# MODELNET40 Example with LightConvPoint

# other imports
import numpy as np
import os
from tqdm import tqdm
import argparse
from sklearn.metrics import confusion_matrix
import h5py
import yaml

# torch imports
import torch
import torch.utils.data

from modelnet40_dataset import Modelnet40_dataset as Dataset
import lightconvpoint.utils.metrics as metrics
from lightconvpoint.utils import get_network


def get_data(rootdir, files):

    train_filenames = []
    for line in open(os.path.join(rootdir, files)):
        line = line.split("\n")[0]
        line = os.path.basename(line)
        train_filenames.append(os.path.join(rootdir, line))

    data = []
    labels = []
    for filename in train_filenames:
        f = h5py.File(filename, "r")
        data.append(f["data"])
        labels.append(f["label"])

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)

    return data, labels


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(_config):

    print(_config)
    savedir_root = _config["training"]["savedir"]
    device = torch.device(_config["misc"]["device"])

    # activate cudnn benchmark
    if _config["misc"]["device"] == "cuda":
        torch.backends.cudnn.benchmark = True

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
    net.load_state_dict(
        torch.load(os.path.join(savedir_root, "checkpoint.pth"), map_location=device)[
            "state_dict"
        ]
    )
    net.to(device)
    print("Done")

    print("get the data path...", end="", flush=True)
    rootdir = os.path.join(_config["dataset"]["datasetdir"], _config["dataset"]["dataset"])
    print("done")

    print("Getting test files...", end="", flush=True)
    test_data, test_labels = get_data(rootdir, "test_files.txt")
    print("done - ", test_data.shape[0], " test files")

    print("Creating dataloaders...", end="", flush=True)
    ds_test = Dataset(
        test_data,
        test_labels,
        pt_nbr=_config["dataset"]["npoints"],
        training=False,
        network_function=network_function,
        num_iter_per_shape=_config["test"]["num_iter_per_shape"],
    )
    test_loader = torch.utils.data.DataLoader(
        ds_test,
        batch_size=_config["test"]["batchsize"],
        shuffle=False,
        num_workers=_config["misc"]["threads"],
    )
    print("done")

    net.eval()

    cm = np.zeros((N_LABELS, N_LABELS))
    test_oa = "0"
    test_aa = "0"
    test_aiou = "0"
    with torch.no_grad():

        predictions = np.zeros((test_data.shape[0], N_LABELS), dtype=float)
        t = tqdm(test_loader, desc="Test", ncols=100, disable=_config["misc"]["disable_tqdm"])
        for data in t:

            pts = data["pts"]
            features = data["features"]
            targets = data["target"]
            indices = data["index"]
            net_ids = data["net_indices"]
            net_support = data["net_support"]

            features = features.to(device)
            pts = pts.to(device)
            targets = targets.to(device)
            for i in range(len(net_ids)):
                net_ids[i] = net_ids[i].to(device)
            for i in range(len(net_support)):
                net_support[i] = net_support[i].to(device)

            outputs = net(features, pts, support_points=net_support, indices=net_ids)

            outputs_np = outputs.cpu().detach().numpy()

            # fill the predictions per shape
            for i in range(indices.shape[0]):
                predictions[indices[i].item()] += outputs_np[i]

        predictions = np.argmax(predictions, axis=1)
        cm = confusion_matrix(test_labels, predictions, labels=list(range(N_LABELS)))

        # scores
        test_oa = f"{metrics.stats_overall_accuracy(cm)*100:.5f}"
        test_aa = f"{metrics.stats_accuracy_per_class(cm)[0]*100:.5f}"
        test_aiou = f"{metrics.stats_iou_per_class(cm)[0]*100:.5f}"

        print("OA", test_oa)
        print("AA", test_aa)
        print("IoU", test_aiou)


if __name__ == "__main__":

    # get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to config file in savedir")
    args = parser.parse_args()

    # load the configuration
    config = yaml.load(open(args.config))

    config["training"]["savedir"] = os.path.dirname(args.config)

    # call the main function
    main(config)
