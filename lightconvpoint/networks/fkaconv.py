import torch
import torch.nn as nn
from torch.nn.modules import activation
import lightconvpoint.nn as lcp_nn
from lightconvpoint.nn.conv_fkaconv import FKAConv as conv
from lightconvpoint.nn.pool import max_pool
from lightconvpoint.nn.sampling_knn import sampling_knn_quantized as sampling_knn
from lightconvpoint.nn.sampling import sampling_quantized as sampling, sampling_apply_on_data
from lightconvpoint.nn.knn import knn

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.cv0 = nn.Conv1d(in_channels, in_channels//2, 1)
        self.bn0 = nn.BatchNorm1d(in_channels//2)
        self.cv1 = conv(in_channels//2, in_channels//2, kernel_size, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels//2)
        self.cv2 = nn.Conv1d(in_channels//2, out_channels, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def batched_index_select(self, input, dim, index):
        """Gather input with respect to the index tensor."""
        index_shape = index.shape
        views = [input.shape[0]] + [
            1 if i != dim else -1 for i in range(1, len(input.shape))
        ]
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index).view(
            input.size(0), -1, index_shape[1], index_shape[2]
        )

    def forward(self, x, pos, support_points, indices):

        x_short = x

        x = self.activation(self.bn0(self.cv0(x)))
        x = self.activation(self.bn1(self.cv1(x, pos, support_points, indices)))
        x = self.bn2(self.cv2(x))

        if x_short.shape[2] != x.shape[2]:
            x_short = max_pool(x_short, indices)
        x_short = self.shortcut(x_short)

        return self.activation(x + x_short)


class FKAConv(nn.Module):

    def __init__(self, in_channels, out_channels, segmentation=False):
        super().__init__()

        self.segmentation = segmentation

        self.cv0 = conv(in_channels, 64, 16)
        self.bn0 = nn.BatchNorm1d(64)

        hidden = 64

        self.resnetb01 = ResidualBlock(hidden, hidden, 16)
        self.resnetb10 = ResidualBlock(hidden, 2*hidden, 16)
        self.resnetb11 = ResidualBlock(2*hidden, 2*hidden, 16)
        self.resnetb20 = ResidualBlock(2*hidden, 4*hidden, 16)
        self.resnetb21 = ResidualBlock(4*hidden, 4*hidden, 16)
        self.resnetb30 = ResidualBlock(4*hidden, 8*hidden, 16)
        self.resnetb31 = ResidualBlock(8*hidden, 8*hidden, 16)
        self.resnetb40 = ResidualBlock(8*hidden, 16*hidden, 16)
        self.resnetb41 = ResidualBlock(16*hidden, 16*hidden, 16)
        
        if self.segmentation:
            self.cv3d = nn.Conv1d(24*hidden, 8 * hidden, 1)
            self.bn3d = nn.BatchNorm1d(8 * hidden)
            self.cv2d = nn.Conv1d(12 * hidden, 4 * hidden, 1)
            self.bn2d = nn.BatchNorm1d(4 * hidden)
            self.cv1d = nn.Conv1d(6 * hidden, 2 * hidden, 1)
            self.bn1d = nn.BatchNorm1d(2 * hidden)
            self.cv0d = nn.Conv1d(3 * hidden, hidden, 1)
            self.bn0d = nn.BatchNorm1d(hidden)
            self.fcout = nn.Conv1d(hidden, out_channels, 1)
        else:
            self.fcout = nn.Linear(1024, out_channels)

        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ReLU()

    def compute_indices(self, pos):
        
        ids0, _ = sampling_knn(pos, 16, ratio=1)
        ids10_support, support1 = sampling(pos, ratio=0.25, return_support_points=True)
        ids10 = sampling_apply_on_data(ids0, ids10_support)

        ids11, _ = sampling_knn(support1, 16, ratio=1)
        ids20_support, support2 = sampling(support1, ratio=0.25, return_support_points=True)
        ids20 = sampling_apply_on_data(ids11, ids20_support)

        ids21, _ = sampling_knn(support2, 16, ratio=1)
        ids30_support, support3 = sampling(support2, ratio=0.25, return_support_points=True)
        ids30 = sampling_apply_on_data(ids21, ids30_support)

        ids31, _ = sampling_knn(support3, 16, ratio=1)
        ids40_support, support4 = sampling(support3, ratio=0.25, return_support_points=True)
        ids40 = sampling_apply_on_data(ids31, ids40_support)

        ids41, _ = sampling_knn(support4, 16, ratio=1)

        indices = [ids0, ids10, ids11, ids20, ids21, ids30, ids31, ids40, ids41]
        support_points = [support1, support2, support3, support4]

        if self.segmentation:
            ids3u = knn(support4, support3, 1)
            ids2u = knn(support3, support2, 1)
            ids1u = knn(support2, support1, 1)
            ids0u = knn(support1, pos, 1)
            indices = indices + [ids3u, ids2u, ids1u, ids0u]

        return None, indices, support_points


    def forward_with_features(self, x, pos, support_points=None, indices=None):

        if (support_points is None) or (indices is None):
            _, indices, support_points = self.compute_indices(pos)

        if self.segmentation:
            ids0, ids10, ids11, ids20, ids21, ids30, ids31, ids40, ids41, ids3u, ids2u, ids1u, ids0u = indices
        else:
            ids0, ids10, ids11, ids20, ids21, ids30, ids31, ids40, ids41 = indices
        support1, support2, support3, support4 = support_points

        x0 = self.activation(self.bn0(self.cv0(x, pos, pos, ids0)))
        x0 = self.resnetb01(x0, pos, pos, ids0)
        x1 = self.resnetb10(x0, pos, support1, ids10)
        x1 = self.resnetb11(x1, support1, support1, ids11)
        x2 = self.resnetb20(x1, support1, support2, ids20)
        x2 = self.resnetb21(x2, support2, support2, ids21)
        x3 = self.resnetb30(x2, support2, support3, ids30)
        x3 = self.resnetb31(x3, support3, support3, ids31)
        x4 = self.resnetb40(x3, support3, support4, ids40)
        x4 = self.resnetb41(x4, support4, support4, ids41)

        if self.segmentation:
            xout = sampling_apply_on_data(x4, ids3u, dim=2)
            xout = self.activation(self.bn3d(self.cv3d(torch.cat([xout, x3], dim=1))))
            xout = sampling_apply_on_data(xout, ids2u, dim=2)
            xout = self.activation(self.bn2d(self.cv2d(torch.cat([xout, x2], dim=1))))
            xout = sampling_apply_on_data(xout, ids1u, dim=2)
            xout = self.activation(self.bn1d(self.cv1d(torch.cat([xout, x1], dim=1))))
            xout = sampling_apply_on_data(xout, ids0u, dim=2)
            xout = self.activation(self.bn0d(self.cv0d(torch.cat([xout, x0], dim=1))))
            xout = self.fcout(xout)
        else:
            xout = x4.mean(dim=2)
            xout = self.dropout(xout)
            xout = self.fcout(xout)

        return xout

    def forward(self, x, pos, support_points=None, indices=None):
        if x is None:
            return self.compute_indices(pos)
        else:
            return self.forward_with_features(x, pos, support_points, indices)