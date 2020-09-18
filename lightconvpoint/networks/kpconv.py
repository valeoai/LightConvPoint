import torch
import torch.nn as nn
import lightconvpoint.nn as lcp_nn


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        K,
        ConvNet,
        Search,
        stride=1,
        npoints=-1,
    ):
        super().__init__()

        self.cv0 = nn.Conv1d(in_channels, in_channels // 2, 1)
        self.cv1 = lcp_nn.Conv(
            ConvNet(in_channels // 2, in_channels // 2, kernel_size),
            Search(K=K, stride=stride, npoints=npoints),
            activation=nn.ReLU(),
            normalization=nn.BatchNorm1d(in_channels // 2),
        )
        self.cv2 = nn.Conv1d(in_channels // 2, out_channels, 1)

        self.bn0 = nn.BatchNorm1d(in_channels // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.short = (
            nn.Conv1d(in_channels, out_channels, 1)
            if out_channels != in_channels
            else nn.Identity()
        )
        self.bn_short = (
            nn.BatchNorm1d(out_channels)
            if out_channels != in_channels
            else nn.Identity()
        )
        self.short_pool = (
            lcp_nn.MaxPool() if (stride > 1) or (npoints > 0) else lcp_nn.Identity()
        )

        self.relu = nn.ReLU()

    def forward(self, x, input_pts, support_points=None, indices=None):

        shortcut = x

        x = self.relu(self.bn0(self.cv0(x))) if x is not None else None
        x, pts, ids = self.cv1(x, input_pts, support_points, indices=indices)

        if x is not None:
            x = self.bn2(self.cv2(x))
            shortcut, _, _ = self.short_pool(
                self.bn_short(self.short(shortcut)), input_pts, pts, indices=ids
            )
            return self.relu(x + shortcut), pts, ids
        else:
            return None, pts, ids


class KPConvCls(nn.Module):
    """KPConv classification network.

    Network inspired from the KPConv paper and code
    (https://github.com/HuguesTHOMAS/KPConv)

    # Arguments
        in_channels: int.
            The number of input channels.
        out_channels: int.
            The number of output  channels.
        ConvNet: convolutional layer.
            The convolutional class to be used in the network.
        Search: search algorithm.
            The search class to be used in the network.
    """

    def __init__(self, in_channels, out_channels, ConvNet, Search, **kwargs):
        super().__init__()

        pl = 64
        kernel_size = 16
        K = 16

        # Encoder
        self.cv0 = lcp_nn.Conv(
            ConvNet(in_channels, pl, kernel_size),
            Search(K=K),
            activation=nn.ReLU(),
            normalization=nn.BatchNorm1d(pl),
        )

        self.resnetb01 = ResnetBlock(pl, pl, kernel_size, K, ConvNet, Search)
        self.resnetb10 = ResnetBlock(
            pl, 2 * pl, kernel_size, K, ConvNet, Search, npoints=512
        )
        self.resnetb11 = ResnetBlock(2 * pl, 2 * pl, kernel_size, K, ConvNet, Search)
        self.resnetb20 = ResnetBlock(
            2 * pl, 4 * pl, kernel_size, K, ConvNet, Search, npoints=128
        )
        self.resnetb21 = ResnetBlock(4 * pl, 4 * pl, kernel_size, K, ConvNet, Search)
        self.resnetb30 = ResnetBlock(
            4 * pl, 8 * pl, kernel_size, K, ConvNet, Search, npoints=32
        )
        self.resnetb31 = ResnetBlock(8 * pl, 8 * pl, kernel_size, K, ConvNet, Search)
        self.resnetb40 = ResnetBlock(
            8 * pl, 16 * pl, kernel_size, K, ConvNet, Search, npoints=8
        )
        self.resnetb41 = ResnetBlock(16 * pl, 16 * pl, kernel_size, K, ConvNet, Search)

        self.fc = nn.Conv1d(16 * pl, out_channels, 1)
        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(
        self, x, input_pts, support_points=None, indices=None, return_features=False
    ):

        if support_points is None:
            support_points = [None for _ in range(14)]
        if indices is None:
            indices = [None for _ in range(14)]        

        # ENCODER
        x0, _, ids00 = self.cv0(x, input_pts, input_pts, indices=indices[0])
        x0, _, ids01 = self.resnetb01(x0, input_pts, input_pts, indices=indices[1])
        x1, pts1, ids10 = self.resnetb10(
            x0, input_pts, support_points[0], indices=indices[2]
        )
        x1, _, ids11 = self.resnetb11(x1, pts1, pts1, indices=indices[3])
        x2, pts2, ids20 = self.resnetb20(
            x1, pts1, support_points[1], indices=indices[4]
        )
        x2, _, ids21 = self.resnetb21(x2, pts2, pts2, indices=indices[5])
        x3, pts3, ids30 = self.resnetb30(
            x2, pts2, support_points[2], indices=indices[6]
        )
        x3, _, ids31 = self.resnetb31(x3, pts3, pts3, indices=indices[7])
        x4, pts4, ids40 = self.resnetb40(
            x3, pts3, support_points[3], indices=indices[8]
        )
        x4, _, ids41 = self.resnetb41(x4, pts4, pts4, indices=indices[9])

        if x4 is not None:
            xout = self.drop(x4)
            xout = self.fc(xout)

            return xout.mean(dim=2)
        else:
            return (
                None,
                [ids00, ids01, ids10, ids11, ids20, ids21, ids30, ids31, ids40, ids41],
                [pts1, pts2, pts3, pts4],
            )


class KPConvSeg(nn.Module):

    """KPConv segmentation network.

    Network inspired from the KPConv paper and code
    (https://github.com/HuguesTHOMAS/KPConv)

    # Arguments
        in_channels: int.
            The number of input channels.
        out_channels: int.
            The number of output  channels.
        ConvNet: convolutional layer.
            The convolutional class to be used in the network.
        Search: search algorithm.
            The search class to be used in the network.
    """

    def __init__(self, in_channels, out_channels, ConvNet, Search, **kwargs):
        super().__init__()

        pl = 64
        kernel_size = 16
        K = 16

        # Encoder
        self.cv0 = lcp_nn.Conv(
            ConvNet(in_channels, pl, kernel_size),
            Search(K=K),
            activation=nn.ReLU(),
            normalization=nn.BatchNorm1d(pl),
        )
        self.resnetb01 = ResnetBlock(pl, pl, kernel_size, K, ConvNet, Search)
        self.resnetb10 = ResnetBlock(
            pl, 2 * pl, kernel_size, K, ConvNet, Search, npoints=512
        )
        self.resnetb11 = ResnetBlock(2 * pl, 2 * pl, kernel_size, K, ConvNet, Search)
        self.resnetb20 = ResnetBlock(
            2 * pl, 4 * pl, kernel_size, K, ConvNet, Search, npoints=128
        )
        self.resnetb21 = ResnetBlock(4 * pl, 4 * pl, kernel_size, K, ConvNet, Search)
        self.resnetb30 = ResnetBlock(
            4 * pl, 8 * pl, kernel_size, K, ConvNet, Search, npoints=32
        )
        self.resnetb31 = ResnetBlock(8 * pl, 8 * pl, kernel_size, K, ConvNet, Search)
        self.resnetb40 = ResnetBlock(
            8 * pl, 16 * pl, kernel_size, K, ConvNet, Search, npoints=8
        )
        self.resnetb41 = ResnetBlock(16 * pl, 16 * pl, kernel_size, K, ConvNet, Search)

        # Decoder
        self.upsample = lcp_nn.UpSampleNearest()

        self.cv3d = nn.Conv1d(24 * pl, 8 * pl, 1)
        self.cv2d = nn.Conv1d(12 * pl, 4 * pl, 1)
        self.cv1d = nn.Conv1d(6 * pl, 2 * pl, 1)
        self.cv0d = nn.Conv1d(3 * pl, pl, 1)

        self.fc = nn.Conv1d(pl, out_channels, 1)

        self.bn3d = nn.BatchNorm1d(8 * pl)
        self.bn2d = nn.BatchNorm1d(4 * pl)
        self.bn1d = nn.BatchNorm1d(2 * pl)
        self.bn0d = nn.BatchNorm1d(pl)

        self.drop = nn.Dropout(0.5)
        self.features_out_size = pl
        self.relu = nn.ReLU()

    def forward(
        self, x, input_pts, support_points=None, indices=None, return_features=False
    ):

        if support_points is None:
            support_points = [None for _ in range(14)]
        if indices is None:
            indices = [None for _ in range(14)]

        # ENCODER
        x0, _, ids00 = self.cv0(x, input_pts, input_pts, indices=indices[0])
        x0, _, ids01 = self.resnetb01(x0, input_pts, input_pts, indices=indices[1])
        x1, pts1, ids10 = self.resnetb10(
            x0, input_pts, support_points[0], indices=indices[2]
        )
        x1, _, ids11 = self.resnetb11(x1, pts1, pts1, indices=indices[3])
        x2, pts2, ids20 = self.resnetb20(
            x1, pts1, support_points[1], indices=indices[4]
        )
        x2, _, ids21 = self.resnetb21(x2, pts2, pts2, indices=indices[5])
        x3, pts3, ids30 = self.resnetb30(
            x2, pts2, support_points[2], indices=indices[6]
        )
        x3, _, ids31 = self.resnetb31(x3, pts3, pts3, indices=indices[7])
        x4, pts4, ids40 = self.resnetb40(
            x3, pts3, support_points[3], indices=indices[8]
        )
        x4, _, ids41 = self.resnetb41(x4, pts4, pts4, indices=indices[9])

        # DECODER
        x3d, _, ids3u = self.upsample(x4, pts4, pts3, indices=indices[10])
        if x3d is not None:
            x3d = torch.cat([x3d, x3], dim=1)
            x3d = self.relu(self.bn3d(self.cv3d(x3d)))

        x2d, _, ids2u = self.upsample(x3d, pts3, pts2, indices=indices[11])
        if x2d is not None:
            x2d = torch.cat([x2d, x2], dim=1)
            x2d = self.relu(self.bn2d(self.cv2d(x2d)))

        x1d, _, ids1u = self.upsample(x2d, pts2, pts1, indices=indices[12])
        if x1d is not None:
            x1d = torch.cat([x1d, x1], dim=1)
            x1d = self.relu(self.bn1d(self.cv1d(x1d)))

        x0d, _, ids0u = self.upsample(x1d, pts1, input_pts, indices=indices[13])
        if x0d is not None:
            x0d = torch.cat([x0d, x0], dim=1)
            x0d = self.relu(self.bn0d(self.cv0d(x0d)))
            x0d = self.drop(x0d)
            xout = self.fc(x0d)

            if return_features:
                return xout, x0d
            else:
                return xout
        else:
            return (
                None,
                [
                    ids00,
                    ids01,
                    ids10,
                    ids11,
                    ids20,
                    ids21,
                    ids30,
                    ids31,
                    ids40,
                    ids41,
                    ids3u,
                    ids2u,
                    ids1u,
                    ids0u,
                ],
                [pts1, pts2, pts3, pts4],
            )
