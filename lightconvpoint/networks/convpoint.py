import torch
import torch.nn as nn
import lightconvpoint.nn as lcp_nn

# This mdoels for classification and segmentation
# are inspired from ConvPoint
# https://github.com/aboulch/ConvPoint


class ConvPointCls(nn.Module):
    """ConvPoint classification network.

    Network inspired from the KPConv paper and code (https://github.com/aboulch/ConvPoint)

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

        # input 2048
        self.cv1 = lcp_nn.Conv(
            ConvNet(in_channels, 64, 16),
            Search(K=16, npoints=1024),
            activation=nn.ReLU(),
            normalization=nn.BatchNorm1d(64),
        )
        self.cv2 = lcp_nn.Conv(
            ConvNet(64, 128, 16),
            Search(K=16, npoints=256),
            activation=nn.ReLU(),
            normalization=nn.BatchNorm1d(128),
        )
        self.cv3 = lcp_nn.Conv(
            ConvNet(128, 256, 16),
            Search(K=16, npoints=64),
            activation=nn.ReLU(),
            normalization=nn.BatchNorm1d(256),
        )
        self.cv4 = lcp_nn.Conv(
            ConvNet(256, 256, 16),
            Search(K=16, npoints=16),
            activation=nn.ReLU(),
            normalization=nn.BatchNorm1d(256),
        )
        self.cv5 = lcp_nn.Conv(
            ConvNet(256, 512, 16),
            Search(K=16, npoints=1),
            activation=nn.ReLU(),
            normalization=nn.BatchNorm1d(512),
        )

        # last layer
        self.fcout = nn.Linear(512, out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, input_pts, support_points=None, indices=None):

        if support_points is None:
            support_points = [None for _ in range(5)]
        if indices is None:
            indices = [None for _ in range(5)]

        x1, pts1, ids1 = self.cv1(x, input_pts, support_points[0], indices=indices[0])

        x2, pts2, ids2 = self.cv2(x1, pts1, support_points[1], indices=indices[1])

        x3, pts3, ids3 = self.cv3(x2, pts2, support_points[2], indices=indices[2])

        x4, pts4, ids4 = self.cv4(x3, pts3, support_points[3], indices=indices[3])

        x5, pts5, ids5 = self.cv5(x4, pts4, support_points[4], indices=indices[4])

        if x1 is not None:
            xout = x5.view(x5.size(0), -1)
            xout = self.dropout(xout)
            xout = self.fcout(xout)
            return xout
        else:
            return None, [ids1, ids2, ids3, ids4, ids5], [pts1, pts2, pts3, pts4, pts5]


class ConvPointSeg(nn.Module):
    """ConvPoint segmentation network.

    Network inspired from the KPConv paper and code (https://github.com/aboulch/ConvPoint)

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

    def __init__(self, in_channels, out_channels, ConvNet, Search):
        super().__init__()

        # input 8192 / 2048
        self.cv0 = lcp_nn.Conv(ConvNet(in_channels, 64, 16), Search(K=16))  # no stride
        self.cv1 = lcp_nn.Conv(
            ConvNet(64, 64, 16),
            Search(K=16, npoints=2048),
            activation=nn.ReLU(),
            normalization=nn.BatchNorm1d(64),
        )
        self.cv2 = lcp_nn.Conv(
            ConvNet(64, 64, 16),
            Search(K=16, npoints=1024),
            activation=nn.ReLU(),
            normalization=nn.BatchNorm1d(64),
        )
        self.cv3 = lcp_nn.Conv(
            ConvNet(64, 64, 16),
            Search(K=16, npoints=256),
            activation=nn.ReLU(),
            normalization=nn.BatchNorm1d(64),
        )
        self.cv4 = lcp_nn.Conv(
            ConvNet(64, 128, 16),
            Search(K=16, npoints=64),
            activation=nn.ReLU(),
            normalization=nn.BatchNorm1d(128),
        )
        self.cv5 = lcp_nn.Conv(
            ConvNet(128, 128, 16),
            Search(K=16, npoints=16),
            activation=nn.ReLU(),
            normalization=nn.BatchNorm1d(128),
        )
        self.cv6 = lcp_nn.Conv(
            ConvNet(128, 128, 16),
            Search(K=16, npoints=8),
            activation=nn.ReLU(),
            normalization=nn.BatchNorm1d(128),
        )

        self.cv5d = lcp_nn.Conv(
            ConvNet(128, 128, 16),
            Search(K=4),
            activation=nn.ReLU(),
            normalization=nn.BatchNorm1d(128),
        )
        self.cv4d = lcp_nn.Conv(
            ConvNet(256, 128, 16),
            Search(K=4),
            activation=nn.ReLU(),
            normalization=nn.BatchNorm1d(128),
        )
        self.cv3d = lcp_nn.Conv(
            ConvNet(256, 64, 16),
            Search(K=4),
            activation=nn.ReLU(),
            normalization=nn.BatchNorm1d(64),
        )
        self.cv2d = lcp_nn.Conv(
            ConvNet(128, 64, 16),
            Search(K=8),
            activation=nn.ReLU(),
            normalization=nn.BatchNorm1d(64),
        )
        self.cv1d = lcp_nn.Conv(
            ConvNet(128, 64, 16),
            Search(K=8),
            activation=nn.ReLU(),
            normalization=nn.BatchNorm1d(64),
        )
        self.cv0d = lcp_nn.Conv(
            ConvNet(128, 64, 16),
            Search(K=8),
            activation=nn.ReLU(),
            normalization=nn.BatchNorm1d(64),
        )

        self.fcout = nn.Conv1d(128, out_channels, 1)
        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)
        self.features_out_size = 128

    def forward(
        self, x, input_pts, support_points=None, indices=None, return_features=False
    ):

        if support_points is None:
            support_points = [None for _ in range(13)]
        if indices is None:
            indices = [None for _ in range(13)]

        # ENCODER
        x0, pts0, ids0 = self.cv0(x, input_pts, input_pts, indices=indices[0])
        x1, pts1, ids1 = self.cv1(x0, pts0, support_points[0], indices=indices[1])
        x2, pts2, ids2 = self.cv2(x1, pts1, support_points[1], indices=indices[2])
        x3, pts3, ids3 = self.cv3(x2, pts2, support_points[2], indices=indices[3])
        x4, pts4, ids4 = self.cv4(x3, pts3, support_points[3], indices=indices[4])
        x5, pts5, ids5 = self.cv5(x4, pts4, support_points[4], indices=indices[5])
        x6, pts6, ids6 = self.cv6(x5, pts5, support_points[5], indices=indices[6])

        # DECODER
        x5d, _, ids5d = self.cv5d(x6, pts6, pts5, indices=indices[7])
        x5d = torch.cat([x5d, x5], dim=1) if x5d is not None else None
        x4d, _, ids4d = self.cv4d(x5d, pts5, pts4, indices=indices[8])
        x4d = torch.cat([x4d, x4], dim=1) if x4d is not None else None
        x3d, _, ids3d = self.cv3d(x4d, pts4, pts3, indices=indices[9])
        x3d = torch.cat([x3d, x3], dim=1) if x3d is not None else None
        x2d, _, ids2d = self.cv2d(x3d, pts3, pts2, indices=indices[10])
        x2d = torch.cat([x2d, x2], dim=1) if x2d is not None else None
        x1d, _, ids1d = self.cv1d(x2d, pts2, pts1, indices=indices[11])
        x1d = torch.cat([x1d, x1], dim=1) if x1d is not None else None
        x0d, _, ids0d = self.cv0d(x1d, pts1, input_pts, indices=indices[12])

        if x0d is not None:
            x0d = torch.cat([x0d, x0], dim=1)
            xout = self.drop(x0d)
            xout = self.fcout(xout)

            if return_features:
                return xout, x0d
            else:
                return xout

        else:
            return (
                None,
                [
                    ids0,
                    ids1,
                    ids2,
                    ids3,
                    ids4,
                    ids5,
                    ids6,
                    ids5d,
                    ids4d,
                    ids3d,
                    ids2d,
                    ids1d,
                    ids0d,
                ],
                [pts1, pts2, pts3, pts4, pts5, pts6],
            )
