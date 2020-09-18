# MODELNET40 Example with LightConvPoint

# other imports
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import h5py

# torch imports
import torch
import torch.nn.functional as F
import torch.utils.data

import lightconvpoint.utils.metrics as metrics
from lightconvpoint.utils import get_network
from lightconvpoint.datasets.modelnet import Modelnet40_normal_resampled, Modelnet40_ply_hdf5_2048
import lightconvpoint.utils.transformations as lcp_transfo

# SACRED
from sacred import Experiment
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.config import save_config_file

SETTINGS.CAPTURE_MODE = "sys"  # for tqdm
ex = Experiment("ModelNet40")
ex.captured_out_filter = apply_backspaces_and_linefeeds  # for tqdm
ex.add_config("modelnet40.yaml")
######

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@ex.automain
def main(_run, _config):

    print(_config)
    savedir_root = _config["training"]["savedir"]
    device = torch.device(_config["misc"]["device"])

    # save the config file in the directory to restore the configuration
    os.makedirs(savedir_root, exist_ok=True)
    save_config_file(eval(str(_config)), os.path.join(savedir_root, "config.yaml"))

    # parameters for training
    N_LABELS = 40
    input_channels = 1

    print("Creating network...", end="", flush=True)

    def network_function():
        return get_network(
            _config["network"]["model"],
            input_channels,
            N_LABELS,
            _config["network"]["backend_conv"],
            _config["network"]["backend_search"],
        )

    net = network_function()
    net.to(device)
    print("Number of parameters", count_parameters(net))

    print("get the data path...", end="", flush=True)
    rootdir = os.path.join(_config["dataset"]["dir"])
    print("done")

    training_transformations = [
        lcp_transfo.UnitBallNormalize(),
        lcp_transfo.RandomSubSample(_config["dataset"]["npoints"]),
        lcp_transfo.NormalPerturbation(sigma=0.01)
    ]
    test_transformations = [
        lcp_transfo.UnitBallNormalize(),
        lcp_transfo.RandomSubSample(_config["dataset"]["npoints"]),
    ]


    print("Creating dataloaders...", end="", flush=True)
    if _config['dataset']['name'] == "Modelnet40_normal_resampled":
        Dataset = Modelnet40_normal_resampled
    elif _config['dataset']['name'] == "Modelnet40_ply_hdf5_2048":
        Dataset = Modelnet40_ply_hdf5_2048
    ds = Dataset(
        rootdir,
        split='training',
        network_function=network_function,
        transformations_points=training_transformations,
    )
    train_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=_config["training"]["batchsize"],
        shuffle=True,
        num_workers=_config["misc"]["threads"],
    )
    ds_test = Dataset(
        rootdir,
        split='test',
        network_function=network_function,
        transformations_points=test_transformations,
    )
    test_loader = torch.utils.data.DataLoader(
        ds_test,
        batch_size=_config["training"]["batchsize"],
        shuffle=False,
        num_workers=_config["misc"]["threads"],
    )
    print("done")

    print("Creating optimizer...", end="")
    optimizer = torch.optim.Adam(net.parameters(), lr=_config["training"]["lr_start"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, _config["training"]["milestones"], gamma=0.5
    )
    print("done")

    def get_data(data):
        
        pts = data["pts"]
        features = data["features"]
        targets = data["target"]
        net_ids = data["net_indices"]
        net_support = data["net_support"]

        features = features.to(device)
        pts = pts.to(device)
        targets = targets.to(device)
        for i in range(len(net_ids)):
            net_ids[i] = net_ids[i].to(device)
        for i in range(len(net_support)):
            net_support[i] = net_support[i].to(device)

        return pts, features, targets, net_ids, net_support

    for epoch in range(_config["training"]["epoch_nbr"]):

        net.train()
        error = 0
        cm = np.zeros((N_LABELS, N_LABELS))

        train_aloss = "0"
        train_oa = "0"
        train_aa = "0"
        train_aiou = "0"

        t = tqdm(
            train_loader,
            desc="Epoch " + str(epoch),
            ncols=130,
            disable=_config["misc"]["disable_tqdm"],
        )
        for data in t:

            pts, features, targets, net_ids, net_support = get_data(data)

            optimizer.zero_grad()
            outputs = net(features, pts, support_points=net_support, indices=net_ids)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

            # compute scores
            output_np = np.argmax(outputs.cpu().detach().numpy(), axis=1)
            target_np = targets.cpu().numpy()
            cm_ = confusion_matrix(
                target_np.ravel(), output_np.ravel(), labels=list(range(N_LABELS))
            )
            cm += cm_
            error += loss.item()

            # point wise scores on training
            train_oa = "{:.5f}".format(metrics.stats_overall_accuracy(cm))
            train_aa = "{:.5f}".format(metrics.stats_accuracy_per_class(cm)[0])
            train_aiou = "{:.5f}".format(metrics.stats_iou_per_class(cm)[0])
            train_aloss = "{:.5e}".format(error / cm.sum())

            t.set_postfix(OA=train_oa, AA=train_aa, AIOU=train_aiou, ALoss=train_aloss)

        net.eval()
        error = 0
        cm = np.zeros((N_LABELS, N_LABELS))
        test_aloss = "0"
        test_oa = "0"
        test_aa = "0"
        test_aiou = "0"
        with torch.no_grad():

            t = tqdm(
                test_loader,
                desc="  Test " + str(epoch),
                ncols=100,
                disable=_config["misc"]["disable_tqdm"],
            )
            for data in t:

                pts, features, targets, net_ids, net_support = get_data(data)

                outputs = net(
                    features, pts, support_points=net_support, indices=net_ids
                )
                loss = F.cross_entropy(outputs, targets)

                outputs_np = outputs.cpu().detach().numpy()
                pred_labels = np.argmax(outputs_np, axis=1)
                cm_ = confusion_matrix(
                    targets.cpu().numpy(), pred_labels, labels=list(range(N_LABELS))
                )
                cm += cm_
                error += loss.item()

                # point-wise scores on testing
                test_oa = "{:.5f}".format(metrics.stats_overall_accuracy(cm))
                test_aa = "{:.5f}".format(metrics.stats_accuracy_per_class(cm)[0])
                test_aiou = "{:.5f}".format(metrics.stats_iou_per_class(cm)[0])
                test_aloss = "{:.5e}".format(error / cm.sum())

                t.set_postfix(OA=test_oa, AA=test_aa, AIOU=test_aiou, ALoss=test_aloss)

        scheduler.step()

        # create the root folder
        os.makedirs(savedir_root, exist_ok=True)

        # save the checkpoint
        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(savedir_root, "checkpoint.pth"),
        )

        # write the logs
        logs = open(os.path.join(savedir_root, "logs.txt"), "a+")
        logs.write(str(epoch) + " ")
        logs.write(train_aloss + " ")
        logs.write(train_oa + " ")
        logs.write(train_aa + " ")
        logs.write(train_aiou + " ")
        logs.write(test_aloss + " ")
        logs.write(test_oa + " ")
        logs.write(test_aa + " ")
        logs.write(test_aiou + "\n")
        logs.flush()
        logs.close()

        # log for Sacred
        _run.log_scalar("trainOA", train_oa, epoch)
        _run.log_scalar("trainAA", train_aa, epoch)
        _run.log_scalar("trainAIoU", train_aiou, epoch)
        _run.log_scalar("trainLoss", train_aloss, epoch)
        _run.log_scalar("testOA", test_oa, epoch)
        _run.log_scalar("testAA", test_aa, epoch)
        _run.log_scalar("testAIoU", test_aiou, epoch)
        _run.log_scalar("testLoss", test_aloss, epoch)
