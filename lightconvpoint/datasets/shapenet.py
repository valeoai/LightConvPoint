import torch
import numpy as np
from lightconvpoint.nn import with_indices_computation_rotation
import lightconvpoint.utils.data_utils as data_utils
import os
import lightconvpoint.utils.transformations as lcp_transfo

def pc_normalize(points):  # from KPConv code

    # Center and rescale point for 1m radius
    pmin = np.min(points, axis=0)
    pmax = np.max(points, axis=0)
    points -= (pmin + pmax) / 2
    scale = np.max(np.linalg.norm(points, axis=1))
    points *= 1.0 / scale

    return points


class ShapeNet_dataset:

    def __init__(
        self,
        rootdir,
        split='training',
        network_function=None,
        transformations_points=[],
        iter_per_shape=1
    ):
        self.rootdir = rootdir
        self.split = split
        self.t_points = transformations_points
        self.num_iter_per_shape = iter_per_shape
        if network_function is not None:
            self.net = network_function()
        else:
            self.net = None

        self.num_classes = 50
        self.label_names = [
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


        self.category_range = []
        count = 0
        for element in self.label_names:
            part_start = count
            count += element[1]
            part_end = count
            self.category_range.append([part_start, part_end])

        if self.split == 'training':
            filelist_train = os.path.join(rootdir, "train_files.txt")
            filelist_val = os.path.join(rootdir, "val_files.txt")
            (
                data_train,
                labels_shape_train,
                data_num_train,
                labels_pts_train,
                _,
            ) = data_utils.load_seg(filelist_train)
            (
                data_val,
                labels_shape_val,
                data_num_val,
                labels_pts_val,
                _,
             ) = data_utils.load_seg(filelist_val)
            self.data = np.concatenate([data_train, data_val], axis=0)
            self.labels_shape = np.concatenate([labels_shape_train, labels_shape_val], axis=0)
            self.data_num = np.concatenate([data_num_train, data_num_val], axis=0)
            self.labels_pts = np.concatenate([labels_pts_train, labels_pts_val], axis=0)


        elif self.split == 'test':
            filelist_test = os.path.join(rootdir, "test_files.txt")
            (
                data_test,
                labels_shape_test,
                data_num_test,
                labels_pts_test,
                _,
            ) = data_utils.load_seg(filelist_test)
            self.data = data_test
            self.labels_shape = labels_shape_test
            self.data_num = data_num_test
            self.labels_pts = labels_pts_test
            self.weights = None # weights are not defined for the test set

    def get_weights(self):
        
        if self.split == 'training':
            frequences = [0 for i in range(len(self.label_names))]
            for i in range(len(self.label_names)):
                frequences[i] += (self.labels_shape == i).sum()
            for i in range(len(self.label_names)):
                frequences[i] /= self.label_names[i][1]
            frequences = np.array(frequences)
            frequences = frequences.mean() / frequences
            repeat_factor = [sh[1] for sh in self.label_names]
            weights = np.repeat(frequences, repeat_factor)
        else:
            weights = None
        return weights

    # def __init__(
    #     self,
    #     data,
    #     data_num,
    #     labels_pts,
    #     labels_shape,
    #     npoints,
    #     num_iter_per_shape=1,
    #     training=False,
    #     network_function=None,
    # ):

    #     self.data = data
    #     self.data_num = data_num
    #     self.labels_pts = labels_pts
    #     self.labels_shape = labels_shape
    #     self.npoints = npoints
    #     self.num_iter_per_shape = num_iter_per_shape
    #     self.training = training
    #     if network_function is not None:
    #         self.net = network_function()
    #     else:
    #         self.net = None

    @with_indices_computation_rotation
    def __getitem__(self, index):

        index = index // self.num_iter_per_shape

        # get the data
        npts = self.data_num[index]
        pts = self.data[index, :npts]
        seg = self.labels_pts[index, :npts]
        label = self.labels_shape[index]

        pts = np.concatenate([pts, np.expand_dims(seg, axis=1)], axis=1)

        for t in self.t_points:
            if isinstance(t, lcp_transfo.RandomSubSample) or isinstance(t, lcp_transfo.FixedSubSample):
                pts, choice = t(pts, return_choice=True)
            else:
                pts = t(pts)
        seg = pts[:,3]
        pts = pts[:,:3]

        # Switch y and z dimensions
        pts = pts[:, [0, 2, 1]]

        pts = torch.from_numpy(pts).float()
        seg = torch.from_numpy(seg).long()

        pts = pts.transpose(0, 1)
        features = torch.ones(1, pts.shape[1]).float()

        return_dict = {
            "pts": pts,
            "features": features,
            "seg": seg,
            "label": label,
            "index": index,
            "choice": choice,
        }

        return return_dict

    def __len__(self):
        return self.data.shape[0] * self.num_iter_per_shape

    def size(self):
        return self.data.shape[0]