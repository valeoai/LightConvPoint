import torch
import torch.nn as nn


class PCCN(nn.Module):
    """PCCN convolution layer (implementation based on the paper)."""

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
        self.use_bias = bias
        self.dim = dim

        # weight matrix
        self.weight = nn.Parameter(
            torch.Tensor(in_channels, out_channels), requires_grad=True
        )
        torch.nn.init.xavier_uniform_(self.weight.data)

        # bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels, 1), requires_grad=True)
            torch.nn.init.zeros_(self.bias.data)

        # MLP
        modules = []
        proj_dim = self.dim
        modules.append(nn.Linear(proj_dim, 16))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(16, 32))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(32, out_channels))
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

        # create the projector
        mat = self.projector(pts.permute(0, 2, 3, 1))

        mat = mat.transpose(2, 3).unsqueeze(4)
        features = torch.matmul(input.permute(0, 2, 3, 1), self.weight)
        features = features.transpose(2, 3).unsqueeze(3)
        features = torch.matmul(features, mat)
        features = features.view(features.shape[:3])
        features = features.transpose(1, 2)

        # add a bias
        if self.use_bias:
            features = features + self.bias

        return features, support_points
