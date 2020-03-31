import numpy as np
import torch
from lightconvpoint.nn import with_indices_computation_rotation


def pc_normalize(points):  # from KPConv code

    # Center and rescale point for 1m radius
    pmin = np.min(points, axis=0)
    pmax = np.max(points, axis=0)
    points -= (pmin + pmax) / 2
    scale = np.max(np.linalg.norm(points, axis=1))
    points *= 1.0 / scale

    return points


class Modelnet40_dataset(torch.utils.data.Dataset):
    """Main Class for Image Folder loader."""

    def __init__(
        self,
        data,
        labels,
        pt_nbr=2048,
        training=True,
        num_iter_per_shape=1,
        jitter=False,
        network_function=None,
    ):
        """Init function."""

        self.data = data
        self.labels = labels
        self.training = training
        self.pt_nbr = pt_nbr
        self.num_iter_per_shape = num_iter_per_shape
        self.jitter = jitter
        if network_function is not None:
            self.net = network_function()
        else:
            self.net = None

    @with_indices_computation_rotation
    def __getitem__(self, index):
        """Get item."""

        index_ = index // self.num_iter_per_shape

        # get the filename
        pts = self.data[index_]
        target = self.labels[index_]

        indices = np.random.choice(pts.shape[0], self.pt_nbr)
        pts = pts[indices]

        # normalize the point cloud
        pts = pc_normalize(pts)

        # Switch y and z dimensions
        pts = pts[:, [0, 2, 1]]

        if self.training:
            if self.jitter:
                pts += 0.01 * np.random.normal(size=pts.shape)

        pts = torch.from_numpy(pts).float()
        pts = pts.transpose(0, 1)
        features = torch.ones(1, pts.shape[1]).float()

        return_dict = {
            "pts": pts,
            "features": features,
            "target": int(target),
            "index": index_,
        }

        return return_dict

    def __len__(self):
        """Length."""
        return self.data.shape[0] * self.num_iter_per_shape
