import torch
import torch.nn as nn
import lightconvpoint.nn as lcp_nn
from lightconvpoint.nn.conv_fkaconv import FKAConv as conv
from lightconvpoint.nn.pool import max_pool
from lightconvpoint.nn.sampling_knn import sampling_knn_quantized as sampling_knn
from lightconvpoint.nn.sampling import sampling_quantized as sampling, sampling_apply_on_data


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

        self.resnetb01 = ResidualBlock(64, 64, 16)
        self.resnetb10 = ResidualBlock(64, 128, 16)
        self.resnetb11 = ResidualBlock(128, 128, 16)
        self.resnetb20 = ResidualBlock(128, 256, 16)
        self.resnetb21 = ResidualBlock(256, 256, 16)
        self.resnetb30 = ResidualBlock(256, 512, 16)
        self.resnetb31 = ResidualBlock(512, 512, 16)
        self.resnetb40 = ResidualBlock(512, 1024, 16)
        self.resnetb41 = ResidualBlock(1024, 1024, 16)

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

        return None, [ids0, ids10, ids11, ids20, ids21, ids30, ids31, ids40, ids41], [support1, support2, support3, support4]


    def forward_with_features(self, x, pos, support_points=None, indices=None):

        if (support_points is None) or (indices is None):
            _, indices, support_points = self.compute_indices(pos)

        ids0, ids10, ids11, ids20, ids21, ids30, ids31, ids40, ids41 = indices
        support1, support2, support3, support4 = support_points

        x = self.activation(self.bn0(self.cv0(x, pos, pos, ids0)))
        x = self.resnetb01(x, pos, pos, ids0)
        x = self.resnetb10(x, pos, support1, ids10)
        x = self.resnetb11(x, support1, support1, ids11)
        x = self.resnetb20(x, support1, support2, ids20)
        x = self.resnetb21(x, support2, support2, ids21)
        x = self.resnetb30(x, support2, support3, ids30)
        x = self.resnetb31(x, support3, support3, ids31)
        x = self.resnetb40(x, support3, support4, ids40)
        x = self.resnetb41(x, support4, support4, ids41)
        x = x.mean(dim=2)
        x = self.dropout(x)
        x = self.fcout(x)

        return x

    def forward(self, x, pos, support_points=None, indices=None):
        if x is None:
            return self.compute_indices(pos)
        else:
            return self.forward_with_features(x, pos, support_points, indices)