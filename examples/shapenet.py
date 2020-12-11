import os
import numpy as np
from torch_geometric.transforms.sample_points import SamplePoints
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F

from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T

from lightconvpoint.datasets.dataset import get_dataset
from lightconvpoint.utils import get_network
from lightconvpoint.networks.pointnet import Pointnet, ResidualPointnet
# from lightconvpoint.networks.dev_convpoint import ConvPoint
from lightconvpoint.networks.fkaconv import FKAConv as Network
import lightconvpoint.utils.metrics as metrics
from lightconvpoint.utils.misc import wblue, wgreen

path = "/root/no_backup/data/ShapeNet"

Dataset = get_dataset(ShapeNet)
NLABELS = 50



def network_function():
    return Network(3, NLABELS, segmentation=True)

category = None # 'Airplane'  # Pass in `None` to train on all categories.
transform = T.Compose([
    T.FixedPoints(2048),
    T.RandomTranslate(0.01),
    T.RandomRotate(15, axis=0),
    T.RandomRotate(15, axis=1),
    T.RandomRotate(15, axis=2)
])
pre_transform = T.NormalizeScale()

train_dataset = Dataset(path, category, split='trainval', transform=transform,
                         pre_transform=pre_transform, network_function=network_function)
test_dataset = Dataset(path, category, split='test', transform=transform,
                        pre_transform=pre_transform, network_function=network_function)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=6)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=6)

device = torch.device("cuda")
net = network_function()
net.to(device)

print("Creating optimizer...", end="")
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
print("done")


def get_data(data):

    pts = data["pos"].to(device)
    features = data["x"].to(device)
    seg = data["y"].to(device)
    labels = data["category"]
    net_ids = data["net_indices"]
    net_pts = data["net_support"]
    for i in range(len(net_ids)):
        net_ids[i] = net_ids[i].to(device)
    for i in range(len(net_pts)):
        net_pts[i] = net_pts[i].to(device)

    return pts, features, seg, labels, net_ids, net_pts


for epoch in range(30):

    net.train()
    error = 0
    cm = np.zeros((NLABELS, NLABELS))

    train_aloss = "0"
    train_oa = "0"
    train_aa = "0"
    train_aiou = "0"

    t = tqdm(
        train_loader,
        desc="Epoch " + str(epoch),
        ncols=100,
        disable=False,
    )
    for data in t:

        pos, x, y, categories, net_ids, net_pts = get_data(data)

        optimizer.zero_grad()
        outputs = net(x, pos, support_points=net_pts, indices=net_ids)
        loss = F.cross_entropy(outputs, y.squeeze(1))
        loss.backward()
        optimizer.step()

        # compute scores
        for i in range(outputs.shape[0]):
            outs = outputs[i]
            mask = train_dataset.y_mask[categories[i]].squeeze(0).unsqueeze(1).expand_as(outs)
            outs[~mask] = -1e7
            outputs[i] = outs



        output_np = np.argmax(outputs.cpu().detach().numpy(), axis=1)
        target_np = y.cpu().numpy()
        cm_ = confusion_matrix(
            target_np.ravel(), output_np.ravel(), labels=list(range(NLABELS))
        )
        cm += cm_
        error += loss.item()

        # point wise scores on training
        train_oa = "{:.5f}".format(metrics.stats_overall_accuracy(cm))
        train_aa = "{:.5f}".format(metrics.stats_accuracy_per_class(cm)[0])
        train_aiou = "{:.5f}".format(metrics.stats_iou_per_class(cm)[0])
        train_aloss = "{:.5e}".format(error / cm.sum())

        t.set_description(wblue(f"OA {train_oa} | AA {train_aa} | IOU {train_aiou} | ALoss {train_aloss}"))
    
    net.eval()
    error = 0
    cm = np.zeros((NLABELS, NLABELS))

    test_aloss = "0"
    test_oa = "0"
    test_aa = "0"
    test_aiou = "0"

    with torch.no_grad():
        t = tqdm(test_loader, desc="Epoch " + str(epoch), ncols=100, disable=False,)
        for data in t:
            
            pos, x, y, categories, net_ids, net_pts = get_data(data)

            outputs = net(x, pos, support_points=net_pts, indices=net_ids)
            loss = F.cross_entropy(outputs, y.squeeze(1))

            # compute scores
            for i in range(outputs.shape[0]):
                outs = outputs[i]
                mask = train_dataset.y_mask[categories[i]].squeeze(0).unsqueeze(1).expand_as(outs)
                outs[~mask] = -1e7
                outputs[i] = outs



            output_np = np.argmax(outputs.cpu().detach().numpy(), axis=1)
            target_np = y.cpu().numpy()
            cm_ = confusion_matrix(
                target_np.ravel(), output_np.ravel(), labels=list(range(NLABELS))
            )
            cm += cm_
            error += loss.item()

            # point wise scores on training
            test_oa = "{:.5f}".format(metrics.stats_overall_accuracy(cm))
            test_aa = "{:.5f}".format(metrics.stats_accuracy_per_class(cm)[0])
            test_aiou = "{:.5f}".format(metrics.stats_iou_per_class(cm)[0])
            test_aloss = "{:.5e}".format(error / cm.sum())

            t.set_description(wgreen(f"OA {test_oa} | AA {test_aa} | IOU {test_aiou} | ALoss {test_aloss}"))