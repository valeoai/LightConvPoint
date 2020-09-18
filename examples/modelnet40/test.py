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

import lightconvpoint.utils.metrics as metrics
from lightconvpoint.utils import get_network
from lightconvpoint.datasets.modelnet import Modelnet40_normal_resampled, Modelnet40_ply_hdf5_2048
import lightconvpoint.utils.transformations as lcp_transfo

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
    net.eval()
    print("Number of parameters", count_parameters(net))
    print("Done")

    print("get the data path...", end="", flush=True)
    rootdir = _config["dataset"]["dir"]
    print("done")

    test_transformations = [
        lcp_transfo.FixedSubSample(_config["dataset"]["npoints"])
    ]

    print("Creating dataloaders...", end="", flush=True)
    if _config['dataset']['name'] == "Modelnet40_normal_resampled":
        Dataset = Modelnet40_normal_resampled
    elif _config['dataset']['name'] == "Modelnet40_ply_hdf5_2048":
        Dataset = Modelnet40_ply_hdf5_2048
    ds_test = Dataset(
        rootdir,
        split='test',
        network_function=network_function,
        transformations_points=test_transformations,
        iter_per_shape=_config['test']['num_iter_per_shape'],
    )
    test_loader = torch.utils.data.DataLoader(
        ds_test,
        batch_size=_config["test"]["batchsize"],
        shuffle=False,
        num_workers=_config["misc"]["threads"],
    )
    print("done")


    def get_data(data):
        
        pts = data["pts"]
        features = data["features"]
        targets = data["target"]
        index = data["index"]
        net_ids = data["net_indices"]
        net_support = data["net_support"]

        features = features.to(device)
        pts = pts.to(device)
        targets = targets.to(device)
        index = index.to(device)
        for i in range(len(net_ids)):
            net_ids[i] = net_ids[i].to(device)
        for i in range(len(net_support)):
            net_support[i] = net_support[i].to(device)

        return pts, features, targets, index, net_ids, net_support

    cm = np.zeros((N_LABELS, N_LABELS))
    test_oa = "0"
    test_aa = "0"
    test_aiou = "0"
    with torch.no_grad():

        predictions = np.zeros((ds_test.size(), N_LABELS), dtype=float)
        t = tqdm(test_loader, desc="Test", ncols=100)
        for data in t:

            pts, features, _, indices, net_ids, net_support = get_data(data)

            outputs = net(features, pts, support_points=net_support, indices=net_ids)

            outputs_np = outputs.cpu().detach().numpy()

            # fill the predictions per shape
            for i in range(indices.shape[0]):
                predictions[indices[i].item()] += outputs_np[i]

        predictions = np.argmax(predictions, axis=1)
        cm = confusion_matrix(ds_test.get_targets(), predictions, labels=list(range(N_LABELS)))

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
