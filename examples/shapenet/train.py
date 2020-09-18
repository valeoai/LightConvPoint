import lightconvpoint.utils.metrics as metrics
import lightconvpoint.utils.data_utils as data_utils
from lightconvpoint.utils import get_network
from lightconvpoint.datasets.shapenet import ShapeNet_dataset as Dataset
import lightconvpoint.utils.transformations as lcp_transfo

# other imports
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# torch imports
import torch
import torch.nn.functional as F
import torch.utils.data

# SACRED
from sacred import Experiment
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.config import save_config_file

SETTINGS.CAPTURE_MODE = "sys"  # for tqdm
ex = Experiment("Shapenet")
ex.captured_out_filter = apply_backspaces_and_linefeeds  # for tqdm
ex.add_config("shapenet.yaml")
######


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@ex.automain
def main(_run, _config):

    print(_config)

    savedir_root = _config["training"]["savedir"]
    device = torch.device(_config["misc"]["device"])

    # save the config file
    os.makedirs(savedir_root, exist_ok=True)
    save_config_file(eval(str(_config)), os.path.join(savedir_root, "config.yaml"))

    print("get the data path...", end="", flush=True)
    rootdir = _config["dataset"]["dir"]
    print("done")

    N_CLASSES = 50

    print("Creating network...", end="", flush=True)

    def network_function():
        return get_network(
            _config["network"]["model"],
            in_channels=1,
            out_channels=N_CLASSES,
            backend_conv=_config["network"]["backend_conv"],
            backend_search=_config["network"]["backend_search"],
        )

    net = network_function()
    net.to(device)
    network_parameters = count_parameters(net)
    print("parameters", network_parameters)

    training_transformations = [
        lcp_transfo.UnitBallNormalize(),
        lcp_transfo.RandomSubSample(_config["dataset"]["npoints"]),
        lcp_transfo.NormalPerturbation(sigma=0.001)
    ]
    test_transformations = [
        lcp_transfo.UnitBallNormalize(),
        lcp_transfo.RandomSubSample(_config["dataset"]["npoints"]),
    ]

    print("Creating dataloader...", end="", flush=True)
    ds = Dataset(
        rootdir,
        'training',
        network_function=network_function,
        transformations_points=training_transformations
    )
    train_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=_config["training"]["batchsize"],
        shuffle=True,
        num_workers=_config["misc"]["threads"],
    )
    ds_test = Dataset(
        rootdir,
        'test',
        network_function=network_function,
        transformations_points=test_transformations
    )
    test_loader = torch.utils.data.DataLoader(
        ds_test,
        batch_size=_config["training"]["batchsize"],
        shuffle=False,
        num_workers=_config["misc"]["threads"],
    )
    print("Done")


    # define weights
    print("Computing weights...", end="", flush=True)
    weights = torch.from_numpy(ds.get_weights()).float().to(device)
    print("Done")

    print("Creating optimizer...", end="", flush=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=_config["training"]["lr_start"], eps=1e-3)
    epoch_start = 0
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        _config["training"]["milestones"],
        gamma=_config["training"]["gamma"],
        last_epoch=epoch_start - 1,
    )
    print("Done")


    def get_data(data):

        pts = data["pts"].to(device)
        features = data["features"].to(device)
        seg = data["seg"].to(device)
        labels = data["label"]
        net_ids = data["net_indices"]
        net_pts = data["net_support"]
        for i in range(len(net_ids)):
            net_ids[i] = net_ids[i].to(device)
        for i in range(len(net_pts)):
            net_pts[i] = net_pts[i].to(device)

        return pts, features, seg, labels, net_ids, net_pts


    # create the log file
    for epoch in range(epoch_start, _config["training"]["epoch_nbr"]):

        # train
        net.train()
        cm = np.zeros((N_CLASSES, N_CLASSES))
        t = tqdm(
            train_loader,
            ncols=120,
            desc=f"Epoch {epoch}",
            disable=_config["misc"]["disable_tqdm"],
        )
        for data in t:

            pts, features, seg, labels, net_ids, net_pts = get_data(data)

            optimizer.zero_grad()
            outputs = net(features, pts, support_points=net_pts, indices=net_ids)
            loss = F.cross_entropy(outputs, seg, weight=weights)
            loss.backward()
            optimizer.step()

            outputs_np = outputs.cpu().detach().numpy()
            for i in range(pts.size(0)):
                # get the number of part for the shape
                object_label = labels[i]
                part_start, part_end = ds.category_range[object_label]

                outputs_np[i, :part_start] = -1e7
                outputs_np[i, part_end:] = -1e7

            output_np = np.argmax(outputs_np, axis=1).copy()
            target_np = seg.cpu().numpy().copy()

            cm_ = confusion_matrix(
                target_np.ravel(), output_np.ravel(), labels=list(range(N_CLASSES))
            )
            cm += cm_

            oa = "{:.3f}".format(metrics.stats_overall_accuracy(cm))
            aa = "{:.3f}".format(metrics.stats_accuracy_per_class(cm)[0])
            iou = "{:.3f}".format(metrics.stats_iou_per_class(cm)[0])

            t.set_postfix(OA=oa, AA=aa, IOU=iou)

        # eval (this is not the final evaluation, see dedicated evaluation)
        net.eval()
        with torch.no_grad():
            cm = np.zeros((N_CLASSES, N_CLASSES))
            t = tqdm(
                test_loader,
                ncols=120,
                desc=f"Test {epoch}",
                disable=_config["misc"]["disable_tqdm"],
            )
            for data in t:

                pts, features, seg, labels, net_ids, net_pts = get_data(data)

                outputs = net(features, pts, support_points=net_pts, indices=net_ids)
                loss = 0

                for i in range(pts.size(0)):
                    # get the number of part for the shape
                    object_label = labels[i]
                    part_start, part_end = ds_test.category_range[object_label]

                    outputs_ = (outputs[i, part_start:part_end]).unsqueeze(0)
                    seg_ = (seg[i] - part_start).unsqueeze(0)

                    loss = loss + weights[object_label] * F.cross_entropy(
                        outputs_, seg_
                    )

                outputs_np = outputs.cpu().detach().numpy()
                for i in range(pts.size(0)):
                    # get the number of part for the shape
                    object_label = labels[i]
                    part_start, part_end = ds_test.category_range[object_label]

                    outputs_np[i, :part_start] = -1e7
                    outputs_np[i, part_end:] = -1e7

                output_np = np.argmax(outputs_np, axis=1).copy()
                target_np = seg.cpu().numpy().copy()

                cm_ = confusion_matrix(
                    target_np.ravel(), output_np.ravel(), labels=list(range(N_CLASSES))
                )
                cm += cm_

                oa_test = "{:.3f}".format(metrics.stats_overall_accuracy(cm))
                aa_test = "{:.3f}".format(metrics.stats_accuracy_per_class(cm)[0])
                iou_test = "{:.3f}".format(metrics.stats_iou_per_class(cm)[0])

                t.set_postfix(OA=oa_test, AA=aa_test, IOU=iou_test)

        # scheduler update
        scheduler.step()

        # save the model
        os.makedirs(savedir_root, exist_ok=True)
        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(savedir_root, "checkpoint.pth"),
        )

        # write the logs
        logs = open(os.path.join(savedir_root, "log.txt"), "a+")
        logs.write(f"{epoch} {oa} {aa} {iou} {oa_test} {aa_test} {iou_test} \n")
        logs.close()

        _run.log_scalar("trainOA", oa, epoch)
        _run.log_scalar("trainAA", aa, epoch)
        _run.log_scalar("trainIoU", iou, epoch)
        _run.log_scalar("testOA", oa_test, epoch)
        _run.log_scalar("testAA", aa_test, epoch)
        _run.log_scalar("testIoU", iou_test, epoch)

    logs.close()
