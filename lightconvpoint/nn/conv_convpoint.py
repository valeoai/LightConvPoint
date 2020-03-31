import torch
import torch.nn as nn
import numpy as np
import math


class ConvPoint(nn.Module):
    """ConvPoint (https://github.com/aboulch/ConvPoint) convolution layer."""

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, dim=3):
        """
        Parameters
        ----------
        in_channels: int
            number of input channels
        out_channels: int
            number of output channels
        kernel_size: int
            number of kernel element
        bias: bool
            add a bias to output features (default is True)
        dim: int
            dimension of the geometrical space (default is 3)
        """
        super().__init__()

        # parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.has_bias = bias
        self.dim = dim

        # Weight
        self.weight = nn.Parameter(
            torch.Tensor(in_channels * self.kernel_size, out_channels),
            requires_grad=True,
        )
        bound = math.sqrt(3.0) * math.sqrt(2.0 / (in_channels + out_channels))
        self.weight.data.uniform_(-bound, bound)

        # bias
        if self.has_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels), requires_grad=True)
            torch.nn.init.zeros_(self.bias.data)

        # centers
        center_data = np.zeros((self.dim, self.kernel_size))
        for i in range(self.kernel_size):
            coord = np.random.rand(self.dim) * 2 - 1
            while (coord ** 2).sum() > 1:
                coord = np.random.rand(self.dim) * 2 - 1
            center_data[:, i] = coord
        self.centers = nn.Parameter(
            torch.from_numpy(center_data).float(), requires_grad=True
        )

        # MLP
        modules = []
        proj_dim = self.dim * self.kernel_size
        for i in range(3):
            modules.append(nn.Linear(proj_dim, self.kernel_size))
            modules.append(nn.ReLU())
            proj_dim = self.kernel_size
        self.projector = nn.Sequential(*modules)

    def normalize_points(self, pts, radius=None):
        maxi = torch.sqrt((pts.detach() ** 2).sum(1).max(2)[0])
        maxi = maxi + (maxi == 0)
        return pts / maxi.view(maxi.size(0), 1, maxi.size(1), 1)

    def forward(self, input, points, support_points):

        # center the neighborhoods (local coordinates)
        pts = points - support_points.unsqueeze(3)

        # normalize points
        pts = self.normalize_points(pts)

        # project features on kernel points
        pts = (pts.permute(0, 2, 3, 1).unsqueeze(4) - self.centers).contiguous()
        pts = pts.view(pts.size(0), pts.size(1), pts.size(2), -1)
        mat = self.projector(pts)
        features = input.transpose(1, 2)
        features = torch.matmul(features, mat)

        # apply kernel weights
        features = torch.matmul(
            features.view(features.size(0), features.size(1), -1), self.weight
        )
        features = features / input.shape[3]

        # apply bias
        if self.has_bias:
            features = features + self.bias

        features = features.transpose(1, 2)
        return features, support_points
