import torch
import torch.nn as nn
import numpy as np


class Conv(nn.Module):

    def __init__(self, network, search,
                 activation=None,
                 normalization=None,
                 ):
        """
        Parameters
        ----------
        network: torch network
            network for computing the convolution
        search: spatial search object
            spatial knn object
        """
        super(Conv, self).__init__()
        self.network = network
        self.search = search
        self.activation = activation
        self.norm = normalization

    def batched_index_select(self, input, dim, index):
        """
        Slicing of input with respect to the index tensor.

        Parameters
        ----------
        input: torch tensor
            tensor to be indexed
        dim: int
            dimension of the index
        index: torch tensor
            indices to gather
        """
        index_shape = index.shape
        views = [input.shape[0]] + \
            [1 if i != dim else -1 for i in range(1, len(input.shape))]
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index).view(input.size(0), -1, index_shape[1], index_shape[2])

    def forward(self, input, points, support_points=None, indices=None):
        """
        Forward function of the layer.
        """

        if indices is None:
            # search the support points and the neighborhoods
            indices, support_points = self.search(points, support_points)

        if input is None:
            # inpuy is None: do not compute features
            return None, support_points, indices
        else:
            # compute the features
            indices = indices.clone()

            # get the features and point coordinates associated with the indices
            pts = self.batched_index_select(
                points, dim=2, index=indices).contiguous()
            features = self.batched_index_select(
                input, dim=2, index=indices).contiguous()

            # predict the features
            features, support_points = self.network(
                features, pts, support_points.contiguous())

            # apply normalization
            if self.norm is not None:
                features = self.norm(features)

            # apply activation
            if self.activation is not None:
                features = self.activation(features)

            return features, support_points, indices
