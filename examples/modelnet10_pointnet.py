import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F

from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T

from lightconvpoint.datasets.dataset import get_dataset
import lightconvpoint.utils.metrics as metrics

from lightconvpoint.networks.pointnet import Pointnet as Network
# from lightconvpoint.networks.pointnet import ResidualPointnet as Network
from lightconvpoint.utils.misc import wblue, wgreen

path = "/root/no_backup/data/ModelNet10"

Dataset = get_dataset(ModelNet)
NLABELS = 10

def network_function():
    return Network(3, NLABELS, 128)

pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)

train_dataset = Dataset(path, '10', True, transform, pre_transform)
test_dataset = Dataset(path, '10', False, transform, pre_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=6)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=6)

device = torch.device("cuda")
net = network_function()
net.to(device)

print("Creating optimizer...", end="")
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
print("done")

def train(desc="", disable_tqdm=False):
    net.train()
    error = 0
    cm = np.zeros((NLABELS, NLABELS))

    aloss = "0"
    oa = "0"
    aa = "0"

    t = tqdm(train_loader, ncols=100, disable=disable_tqdm)
    for data in t:

        pts = data["pos"].to(device)
        targets = data["y"].to(device)

        optimizer.zero_grad()
        outputs = net(pts)
        loss = F.cross_entropy(outputs, targets.squeeze(1))
        loss.backward()
        optimizer.step()

        # compute scores
        output_np = np.argmax(outputs.cpu().detach().numpy(), axis=1)
        target_np = targets.cpu().numpy()
        cm_ = confusion_matrix(
            target_np.ravel(), output_np.ravel(), labels=list(range(NLABELS))
        )
        cm += cm_
        error += loss.item()

        # point wise scores on training
        oa = "{:.5f}".format(metrics.stats_overall_accuracy(cm))
        aa = "{:.5f}".format(metrics.stats_accuracy_per_class(cm)[0])
        aloss = "{:.5e}".format(error / cm.sum())

        t.set_description(wblue(f'{desc} Loss {aloss} | OA {oa} | AA {aa}'))

def test(desc="", disable_tqdm=False):
    net.eval()
    error = 0
    cm = np.zeros((NLABELS, NLABELS))

    aloss = "0"
    oa = "0"
    aa = "0"

    with torch.no_grad():
        t = tqdm(test_loader, ncols=100, disable=disable_tqdm)
        for data in t:

            pts = data["pos"].to(device)
            targets = data["y"].to(device)

            outputs = net(pts)
            loss = F.cross_entropy(outputs, targets.squeeze(1))

            # compute scores
            output_np = np.argmax(outputs.cpu().detach().numpy(), axis=1)
            target_np = targets.cpu().numpy()
            cm_ = confusion_matrix(
                target_np.ravel(), output_np.ravel(), labels=list(range(NLABELS))
            )
            cm += cm_
            error += loss.item()

            # point wise scores on training
            oa = "{:.5f}".format(metrics.stats_overall_accuracy(cm))
            aa = "{:.5f}".format(metrics.stats_accuracy_per_class(cm)[0])
            aloss = "{:.5e}".format(error / cm.sum())

            t.set_description(wgreen(f'{desc} Loss {aloss} | OA {oa} | AA {aa}'))


for epoch in range(30):

    train(desc=f"Epoch {epoch}")
    test(desc=f" Test {epoch}")