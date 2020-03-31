import lightconvpoint.utils.metrics as metrics
import lightconvpoint.utils.data_utils as data_utils
from lightconvpoint.utils import get_network
from shapenet_dataset import ShapeNet_dataset as Dataset

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
######


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Configuration
@ex.config
def my_config():
    datasetdir = None
    dataset = None
    savedir = None
    batchsize = None
    device = None
    npoints = None
    num_iter_per_shape = None
    model = None
    backend_conv = None
    backend_search = None
    lr_start = None
    epoch_nbr = None
    milestones = None
    gamma = None
    use_category = None
    weighting = None
    threads = None
    disable_tqdm = None


######


@ex.automain
def main(_run, _config):

    print(_config)

    savedir_root = _config["savedir"]
    device = torch.device(_config["device"])

    # save the config file
    os.makedirs(savedir_root, exist_ok=True)
    save_config_file(eval(str(_config)), os.path.join(savedir_root, "config.yaml"))

    print("get the data path...", end="", flush=True)
    rootdir = os.path.join(_config["datasetdir"], _config["dataset"])
    print("done")

    filelist_train = os.path.join(rootdir, "train_files.txt")
    filelist_val = os.path.join(rootdir, "val_files.txt")
    filelist_test = os.path.join(rootdir, "test_files.txt")

    N_CLASSES = 50

    shapenet_labels = [
        ["Airplane", 4],
        ["Bag", 2],
        ["Cap", 2],
        ["Car", 4],
        ["Chair", 4],
        ["Earphone", 3],
        ["Guitar", 3],
        ["Knife", 2],
        ["Lamp", 4],
        ["Laptop", 2],
        ["Motorbike", 6],
        ["Mug", 2],
        ["Pistol", 3],
        ["Rocket", 3],
        ["Skateboard", 3],
        ["Table", 3],
    ]
    category_range = []
    count = 0
    for element in shapenet_labels:
        part_start = count
        count += element[1]
        part_end = count
        category_range.append([part_start, part_end])

    # Prepare inputs
    print("Preparing datasets...", end="", flush=True)
    (
        data_train,
        labels_shape_train,
        data_num_train,
        labels_pts_train,
        _,
    ) = data_utils.load_seg(filelist_train)
    data_val, labels_shape_val, data_num_val, labels_pts_val, _ = data_utils.load_seg(
        filelist_val
    )
    (
        data_test,
        labels_shape_test,
        data_num_test,
        labels_pts_test,
        _,
    ) = data_utils.load_seg(filelist_test)
    data_train = np.concatenate([data_train, data_val], axis=0)
    labels_shape_train = np.concatenate([labels_shape_train, labels_shape_val], axis=0)
    data_num_train = np.concatenate([data_num_train, data_num_val], axis=0)
    labels_pts_train = np.concatenate([labels_pts_train, labels_pts_val], axis=0)
    print("Done", data_train.shape)

    # define weights
    print("Computing weights...", end="", flush=True)
    frequences = [0 for i in range(len(shapenet_labels))]
    for i in range(len(shapenet_labels)):
        frequences[i] += (labels_shape_train == i).sum()
    for i in range(len(shapenet_labels)):
        frequences[i] /= shapenet_labels[i][1]
    frequences = np.array(frequences)
    frequences = frequences.mean() / frequences
    repeat_factor = [sh[1] for sh in shapenet_labels]
    frequences = np.repeat(frequences, repeat_factor)
    weights = torch.from_numpy(frequences).float().to(device)
    print("Done")

    print("Creating network...", end="", flush=True)

    def network_function():
        return get_network(
            _config["model"],
            in_channels=1,
            out_channels=N_CLASSES,
            ConvNet_name=_config["backend_conv"],
            Search_name=_config["backend_search"],
        )

    net = network_function()
    net.to(device)
    network_parameters = count_parameters(net)
    print("parameters", network_parameters)

    print("Creating dataloader...", end="", flush=True)
    ds = Dataset(
        data_train,
        data_num_train,
        labels_pts_train,
        labels_shape_train,
        npoints=_config["npoints"],
        training=True,
        network_function=network_function,
    )
    train_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=_config["batchsize"],
        shuffle=True,
        num_workers=_config["threads"],
    )
    ds_test = Dataset(
        data_test,
        data_num_test,
        labels_pts_test,
        labels_shape_test,
        npoints=_config["npoints"],
        training=False,
        network_function=network_function,
    )
    test_loader = torch.utils.data.DataLoader(
        ds_test,
        batch_size=_config["batchsize"],
        shuffle=False,
        num_workers=_config["threads"],
    )
    print("Done")

    print("Creating optimizer...", end="", flush=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=_config["lr_start"], eps=1e-3)
    epoch_start = 0
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        _config["milestones"],
        gamma=_config["gamma"],
        last_epoch=epoch_start - 1,
    )
    print("Done")

    # create the log file
    for epoch in range(epoch_start, _config["epoch_nbr"]):

        # train
        net.train()
        cm = np.zeros((N_CLASSES, N_CLASSES))
        t = tqdm(
            train_loader,
            ncols=120,
            desc=f"Epoch {epoch}",
            disable=_config["disable_tqdm"],
        )
        for data in t:

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

            optimizer.zero_grad()
            outputs = net(features, pts, support_points=net_pts, indices=net_ids)
            loss = F.cross_entropy(outputs, seg, weight=weights)

            loss.backward()
            optimizer.step()

            outputs_np = outputs.cpu().detach().numpy()
            for i in range(pts.size(0)):
                # get the number of part for the shape
                object_label = labels[i]
                part_start, part_end = category_range[object_label]

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
                disable=_config["disable_tqdm"],
            )
            for data in t:
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

                outputs = net(features, pts, support_points=net_pts, indices=net_ids)
                loss = 0

                for i in range(pts.size(0)):
                    # get the number of part for the shape
                    object_label = labels[i]
                    part_start, part_end = category_range[object_label]

                    outputs_ = (outputs[i, part_start:part_end]).unsqueeze(0)
                    seg_ = (seg[i] - part_start).unsqueeze(0)

                    loss = loss + weights[object_label] * F.cross_entropy(
                        outputs_, seg_
                    )

                outputs_np = outputs.cpu().detach().numpy()
                for i in range(pts.size(0)):
                    # get the number of part for the shape
                    object_label = labels[i]
                    part_start, part_end = category_range[object_label]

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
