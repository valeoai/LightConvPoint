import numpy as np
import os
from tqdm import tqdm
import argparse
from sklearn.metrics import confusion_matrix
import yaml

# torch imports
import torch
import torch.utils.data

import lightconvpoint.utils.data_utils as data_utils
import lightconvpoint.utils.metrics as metrics
from shapenet_dataset import ShapeNet_dataset as Dataset
from lightconvpoint.utils import get_network
from lightconvpoint.knn import knn


def nearest_correspondance(pts_src, pts_dest, data_src, K=1):
    pts_src = pts_src.unsqueeze(0).cpu().clone()
    pts_dest = pts_dest.unsqueeze(0).cpu().clone()
    indices = knn(pts_src, pts_dest, K)[0, :, 0]
    if K == 1:
        data_dest = data_src.transpose(0, 1)[indices].transpose(0, 1)
    else:
        # TODO fix that
        data_dest = data_src[indices].mean(1)
    return data_dest


def main(_config):

    print(_config)

    savedir_root = _config["savedir"]
    device = torch.device(_config["device"])

    print("get the data path...", end="", flush=True)
    rootdir = os.path.join(_config["datasetdir"], _config["dataset"])
    print("done")

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
        data_test,
        labels_shape_test,
        data_num_test,
        labels_pts_test,
        _,
    ) = data_utils.load_seg(filelist_test)
    print("Done", data_test.shape)

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
    net.load_state_dict(
        torch.load(os.path.join(savedir_root, "checkpoint.pth"), map_location=device)[
            "state_dict"
        ]
    )
    net.to(device)
    print("Done")

    print("Creating dataloader...", end="", flush=True)
    ds_test = Dataset(
        data_test,
        data_num_test,
        labels_pts_test,
        labels_shape_test,
        npoints=_config["npoints"],
        training=False,
        network_function=network_function,
        num_iter_per_shape=_config["num_iter_per_shape"],
    )
    test_loader = torch.utils.data.DataLoader(
        ds_test,
        batch_size=_config["batchsize"],
        shuffle=False,
        num_workers=_config["threads"],
    )
    print("Done")

    # per shape results
    results = torch.zeros(data_test.shape[0], data_test.shape[1], N_CLASSES)
    results_count = torch.zeros(data_test.shape[0], data_test.shape[1])

    with torch.no_grad():
        cm = np.zeros((N_CLASSES, N_CLASSES))

        t = tqdm(test_loader, ncols=100, desc="Inference")
        for data in t:

            pts = data["pts"].to(device)
            features = data["features"].to(device)
            seg = data["seg"].to(device)
            choices = data["choice"]
            labels = data["label"]
            indices = data["index"]
            net_ids = data["net_indices"]
            net_pts = data["net_support"]
            for i in range(len(net_ids)):
                net_ids[i] = net_ids[i].to(device)
            for i in range(len(net_pts)):
                net_pts[i] = net_pts[i].to(device)

            outputs = net(features, pts, support_points=net_pts, indices=net_ids)

            outputs = outputs.to(torch.device("cpu"))
            for b_id in range(outputs.shape[0]):

                object_label = labels[i]
                part_start, part_end = category_range[object_label]
                outputs[i, :part_start] = -1e7
                outputs[i, part_end:] = -1e7

                shape_id = indices[b_id]
                choice = choices[b_id]

                results_shape = results[shape_id]
                results_shape[choice] += outputs[b_id].transpose(0, 1)
                results[shape_id] = results_shape

                results_count_shape = results_count[shape_id]
                results_count_shape[choice] = 1
                results_count[shape_id] = results_count_shape

            output_np = outputs.cpu().numpy()
            output_np = np.argmax(output_np, axis=1).copy()
            target_np = seg.cpu().numpy().copy()

            cm_ = confusion_matrix(
                target_np.ravel(), output_np.ravel(), labels=list(range(N_CLASSES))
            )
            cm += cm_

    Confs = []
    for s_id in tqdm(range(data_test.shape[0]), ncols=100, desc="Conf. matrices"):

        shape_label = labels_shape_test[s_id]
        # get the number of points
        npts = data_num_test[s_id]

        # get the gt and estimate the number of parts
        label_gt = labels_pts_test[s_id, :npts]
        part_start, part_end = category_range[shape_label]
        label_gt -= part_start

        # get the results
        res_shape = results[s_id, :npts, part_start:part_end]

        # extend results to unseen points
        mask = results_count[s_id, :npts].cpu().numpy() == 1
        if np.logical_not(mask).sum() > 0:
            res_shape_mask = res_shape[mask]
            pts_src = torch.from_numpy(data_test[s_id, :npts][mask]).transpose(0, 1)
            pts_dest = data_test[s_id, :npts]
            pts_dest = pts_dest[np.logical_not(mask)]
            pts_dest = torch.from_numpy(pts_dest).transpose(0, 1)
            res_shape_unseen = nearest_correspondance(
                pts_src, pts_dest, res_shape_mask.transpose(0, 1), K=1
            ).transpose(0, 1)
            res_shape[np.logical_not(mask)] = res_shape_unseen

        res_shape = res_shape.numpy()

        label_pred = np.argmax(res_shape, axis=1)
        cm_shape = confusion_matrix(
            label_gt, label_pred, labels=list(range(part_end - part_start))
        )
        Confs.append(cm_shape)

    # compute IoU per shape
    print("Computing IoUs...", end="", flush=True)
    IoUs_per_shape = []
    for i in range(labels_shape_test.shape[0]):
        IoUs_per_shape.append(metrics.stats_iou_per_class(Confs[i])[0])
    IoUs_per_shape = np.array(IoUs_per_shape)

    # compute object category average
    obj_IoUs = np.zeros(len(shapenet_labels))
    for i in range(len(shapenet_labels)):
        obj_IoUs[i] = IoUs_per_shape[labels_shape_test == i].mean()
    print("Done")

    print(
        "Objs | Inst | Air  Bag  Cap  Car  Cha  Ear  Gui  "
        "Kni  Lam  Lap  Mot  Mug  Pis  Roc  Ska  Tab"
    )
    print(
        "-----|------|-------------------------------"
        "-------------------------------------------------"
    )
    s = "{:3.1f} | {:3.1f} | ".format(
        100 * obj_IoUs.mean(), 100 * np.mean(IoUs_per_shape)
    )
    for AmIoU in obj_IoUs:
        s += "{:3.1f} ".format(100 * AmIoU)
    print(s + "\n")


if __name__ == "__main__":

    # get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to config file in savedir")
    args = parser.parse_args()

    # load the configuration
    config = yaml.load(open(args.config))

    # call the main function
    main(config)
