import torch
import numpy as np
import lightconvpoint.nn
import os
import random
from torchvision import transforms

from lightconvpoint.nn import with_indices_computation_rotation


def pc_normalize(points):  # from KPConv code

    # Center and rescale point for 1m radius
    pmin = np.min(points, axis=0)
    pmax = np.max(points, axis=0)
    points -= (pmin + pmax) / 2
    scale = np.max(np.linalg.norm(points, axis=1))
    points *= 1.0 / scale

    return points


class ShapeNet_dataset():

    def __init__(self, data, data_num, labels_pts, labels_shape, npoints,
                 num_iter_per_shape=1, training=False,
                 network_function=None):

        self.data = data
        self.data_num = data_num
        self.labels_pts = labels_pts
        self.labels_shape = labels_shape
        self.npoints = npoints
        self.num_iter_per_shape = num_iter_per_shape
        self.training = training
        if network_function is not None:
            self.net = network_function()
        else:
            self.net = None

    @with_indices_computation_rotation
    def __getitem__(self, index):

        index = index//self.num_iter_per_shape

        npts = self.data_num[index]
        pts = self.data[index, :npts]
        choice = np.random.choice(npts, self.npoints, replace=True)

        pts = pts[choice]

        # normalize point cloud
        pts = pc_normalize(pts)

        # Switch y and z dimensions
        pts = pts[:, [0, 2, 1]]

        lbs = self.labels_pts[index][choice]
        label_shape = self.labels_shape[index]

        if self.training:
            pts += 0.001 * np.random.normal(size=pts.shape)

        pts = torch.from_numpy(pts).float()
        lbs = torch.from_numpy(lbs).long()

        pts = pts.transpose(0, 1)
        features = torch.ones(1, pts.shape[1]).float()

        return_dict = {"pts": pts, "features": features, "seg": lbs,
                       "label": label_shape, "index": index, "choice": choice}

        return return_dict

    def __len__(self):
        return self.data.shape[0] * self.num_iter_per_shape


# class PartNormalDatasetTest():

#     def __init__ (self, data, data_num, labels_pts, labels_shape, npoints,
#                     num_iter_per_shape=1, training=False,
#                     network_function=None):

#         self.data = data
#         self.data_num = data_num
#         self.labels_pts = labels_pts
#         self.labels_shape = labels_shape
#         self.npoints = npoints
#         self.num_iter_per_shape = num_iter_per_shape
#         self.training = training
#         if network_function is not None:
#             self.net = network_function()
#         else:
#             self.net = None

#     def __getitem__(self, index):

#         index = index//self.num_iter_per_shape

#         npts = self.data_num[index]
#         pts = self.data[index, :npts]
#         choice = np.random.choice(npts, self.npoints, replace=True)

#         pts = pts[choice]

#         # normalize point cloud
#         pts = pc_normalize(pts)

#         # Switch y and z dimensions
#         pts = pts[:, [0, 2, 1]]

#         lbs = self.labels_pts[index][choice]
#         label_shape = self.labels_shape[index]

#         if self.training:
#             pts += 0.001 * np.random.normal(size = pts.shape)

#         pts = torch.from_numpy(pts).float()
#         lbs = torch.from_numpy(lbs).long()

#         pts = pts.transpose(0,1)
#         features = torch.ones(1, pts.shape[1]).float()

#         if self.net is not None:

#             # indices, indices_pts = self.net(None, pts.unsqueeze(0), fast_forward=True)

#             # random rotation
#             pts_r, rotation_matrix = rotate_point_cloud(pts.numpy())
#             pts_r = torch.from_numpy(pts_r).float()
#             indices, indices_pts = self.net(None, pts_r.unsqueeze(0), fast_forward=True)

#             for i in range(len(indices)):
#                 indices[i] = indices[i].squeeze(0)
#             for i in range(len(indices_pts)):
#                 indices_pts[i] = indices_pts[i].squeeze(0)

#                 # inverse rotation
#                 pts_tmp, _ = rotate_point_cloud(indices_pts[i].numpy(), rotation_matrix, inverse=True)
#                 indices_pts[i] = torch.from_numpy(pts_tmp).float()

#             return_data = [pts, features, lbs, label_shape, index, choice]
#             return_data += [len(indices)] + indices + indices_pts
#             return return_data
#         else:
#             return pts, features, lbs, label_shape, index, choice

#     def __len__(self):
#         return self.data.shape[0] * self.num_iter_per_shape
