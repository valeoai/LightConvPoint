import numpy as np
import torch
from lightconvpoint.nn import with_indices_computation_rotation
import os
from tqdm import tqdm
import pandas
import h5py

def pc_normalize(points):  # from KPConv code

    # Center and rescale point for 1m radius
    pmin = np.min(points, axis=0)
    pmax = np.max(points, axis=0)
    points -= (pmin + pmax) / 2
    scale = np.max(np.linalg.norm(points, axis=1))
    points *= 1.0 / scale

    return points


class Modelnet40_ply_hdf5_2048(torch.utils.data.Dataset):
    """Main Class for Image Folder loader."""

    def get_data(self,files):

        train_filenames = []
        for line in open(os.path.join(self.rootdir, files)):
            line = line.split("\n")[0]
            line = os.path.basename(line)
            train_filenames.append(os.path.join(self.rootdir, line))

        data = []
        labels = []
        for filename in train_filenames:
            f = h5py.File(filename, "r")
            data.append(f["data"])
            labels.append(f["label"])

        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)

        return data, labels


    def __init__(
                    self,
                    rootdir,
                    split='training',
                    network_function=None,
                    transformations_points=[],
                    iter_per_shape=1):
        """Init function."""
        self.rootdir = rootdir
        self.split = split
        self.t_points = transformations_points
        self.iter = iter_per_shape

        # get the data
        if self.split == 'training':
            self.data, self.labels = self.get_data("train_files.txt")
        elif self.split == 'test':
            self.data, self.labels = self.get_data("test_files.txt")
        
        if network_function is not None:
            self.net = network_function()
        else:
            self.net = None

    @with_indices_computation_rotation
    def __getitem__(self, index):
        """Get item."""

        # compute index of the shape (modulo the number of iteration per shape)
        index_ = index // self.iter

        # the points and target
        pts = self.data[index_]
        target = self.labels[index_]

        for t in self.t_points:
            pts = t(pts)

        # Switch y and z dimensions
        pts = pts[:, [0, 2, 1]]

        # convert to torch
        pts = torch.from_numpy(pts).float()
        pts = pts.transpose(0, 1)

        # generate features
        features = torch.ones(1, pts.shape[1]).float()

        return_dict = {
            "pts": pts,
            "features": features,
            "target": int(target),
            "index": index_,
        }

        return return_dict

    def size(self):
        return self.data.shape[0]

    def __len__(self):
        """Length."""
        return self.size() * self.iter

    def get_targets(self):
        return self.labels


class Modelnet40_normal_resampled(torch.utils.data.Dataset):
    """Main Class for Image Folder loader."""


    def __init__(
                    self,
                    rootdir,
                    split='training',
                    network_function=None,
                    transformations_points=None,
                    iter_per_shape=1):
        self.rootdir = rootdir
        self.split = split
        self.t_points = transformations_points
        self.iter = iter_per_shape

        dataset = "modelnet40"

        if network_function is not None:
            self.net = network_function()
        else:
            self.net = None

        if dataset=="modelnet10":
            self.catfile = os.path.join(rootdir, 'modelnet10_shape_names.txt')
        elif dataset=="modelnet40":
            self.catfile = os.path.join(rootdir, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))  

        # get the filepath
        shape_ids = {}
        if dataset=="modelnet10":
            shape_ids['training'] = [line.rstrip() for line in open(os.path.join(rootdir, 'modelnet10_train.txt'))] 
            shape_ids['test']= [line.rstrip() for line in open(os.path.join(rootdir, 'modelnet10_test.txt'))]
        else:
            shape_ids['training'] = [line.rstrip() for line in open(os.path.join(rootdir, 'modelnet40_train.txt'))] 
            shape_ids['test']= [line.rstrip() for line in open(os.path.join(rootdir, 'modelnet40_test.txt'))]
        assert(split=='training' or split=='test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(rootdir, shape_names[i], shape_ids[split][i])+'.txt') for i in range(len(shape_ids[split]))]

    def size(self):
        return len(self.datapath)

    def __len__(self):
        """Length."""
        return self.size() * self.iter

    def get_targets(self):
        targets = []
        for d in self.datapath:
            targets.append(self.cat.index(d[0]))
        return np.array(targets, dtype=np.int64)

    @with_indices_computation_rotation
    def __getitem__(self, index):
        """Get item."""

        # compute index of the shape (modulo the number of iteration per shape)
        index_ = index // self.iter

        # get the target
        target = self.cat.index(self.datapath[index_][0])
        data = pandas.read_csv(self.datapath[index_][1], header=0).values.astype(np.float32)

        pts = data[:,:3]
        # normals = data[:,3:] # normals are not used in the dataloader

        if self.t_points is not None:
            for t in self.t_points:
                pts = t(pts)

        # Switch y and z dimensions
        pts = pts[:, [0, 2, 1]]

        # convert to torch
        pts = torch.from_numpy(pts).float()
        pts = pts.transpose(0, 1)

        # generate features
        features = torch.ones(1, pts.shape[1]).float()

        return_dict = {
            "pts": pts,
            "features": features,
            "target": int(target),
            "index": index_,
        }

        return return_dict