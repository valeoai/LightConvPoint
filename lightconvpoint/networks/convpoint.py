import torch
import torch.nn as nn
import lightconvpoint.nn as lcp_nn
from lightconvpoint.nn.conv_convpoint import ConvPoint as conv
from lightconvpoint.nn.sampling_knn import sampling_knn_convpoint as sampling_knn


class ConvPoint(nn.Module):

    def __init__(self, in_channels, out_channels, segmentation=False):
        super().__init__()

        self.segmentation = segmentation

        if self.segmentation:
            
            self.cv0 = conv(in_channels, 64, 16, bias=False)
            self.bn0 = nn.BatchNorm1d(64)      
            self.cv1 = conv(64, 64, 16, bias=False)
            self.bn1 = nn.BatchNorm1d(64)      
            self.cv2 = conv(64, 64, 16, bias=False)
            self.bn2 = nn.BatchNorm1d(64)      
            self.cv3 = conv(64, 64, 16, bias=False)
            self.bn3 = nn.BatchNorm1d(64)      
            self.cv4 = conv(64, 128, 16, bias=False)
            self.bn4 = nn.BatchNorm1d(128)      
            self.cv5 = conv(128, 128, 16, bias=False)
            self.bn5 = nn.BatchNorm1d(128)     
            self.cv6 = conv(128, 128, 16, bias=False)
            self.bn6 = nn.BatchNorm1d(128)     
            self.cv5d = conv(128, 128, 16, bias=False)
            self.bn5d = nn.BatchNorm1d(128)     
            self.cv4d = conv(256, 128, 16, bias=False)
            self.bn4d = nn.BatchNorm1d(128)     
            self.cv3d = conv(256, 64, 16, bias=False)
            self.bn3d = nn.BatchNorm1d(64)      
            self.cv2d = conv(128, 64, 16, bias=False)
            self.bn2d = nn.BatchNorm1d(64)
            self.cv1d = conv(128, 64, 16, bias=False)
            self.bn1d = nn.BatchNorm1d(64)
            self.cv0d = conv(128, 64, 16, bias=False)
            self.bn0d = nn.BatchNorm1d(64)
            self.fcout = nn.Conv1d(128, out_channels, 1)

        else:

            self.cv1 = conv(in_channels, 64, 16, bias=False)
            self.bn1 = nn.BatchNorm1d(64)            
            self.cv2 = conv(64, 128, 16, bias=False)
            self.bn2 = nn.BatchNorm1d(128)
            self.cv3 = conv(128, 256, 16, bias=False)
            self.bn3 = nn.BatchNorm1d(256)
            self.cv4 = conv(256, 256, 16, bias=False)
            self.bn4 = nn.BatchNorm1d(256)
            self.cv5 = conv(256, 512, 16, bias=False)
            self.bn5 = nn.BatchNorm1d(512)
            self.fcout = nn.Linear(512, out_channels)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)


    def compute_indices(self, pos):
        if self.segmentation:
            ids0, support0 = sampling_knn(pos, 16, ratio=1)
            ids1, support1 = sampling_knn(support0, 16, ratio=0.25)
            ids2, support2 = sampling_knn(support1, 16, ratio=0.5)
            ids3, support3 = sampling_knn(support2, 16, ratio=0.25)
            ids4, support4 = sampling_knn(support3, 16, ratio=0.25)
            ids5, support5 = sampling_knn(support4, 16, ratio=0.25)
            ids6, support6 = sampling_knn(support5, 16, ratio=0.25)
            ids5d, _ = sampling_knn(support6, 4, support_points=support5)
            ids4d, _ = sampling_knn(support5, 4, support_points=support4)
            ids3d, _ = sampling_knn(support4, 4, support_points=support3)
            ids2d, _ = sampling_knn(support3, 8, support_points=support2)
            ids1d, _ = sampling_knn(support2, 8, support_points=support1)
            ids0d, _ = sampling_knn(support1, 8, support_points=support0)
            return None, [ids0, ids1, ids2, ids3, ids4, ids5, ids6, ids5d, ids4d, ids3d, ids2d, ids1d, ids0d], [support0, support1, support2, support3, support4, support5, support6]
        else:
            ids1, support1 = sampling_knn(pos, 16, ratio=1)
            ids2, support2 = sampling_knn(support1, 16, ratio=0.25)
            ids3, support3 = sampling_knn(support2, 16, ratio=0.25)
            ids4, support4 = sampling_knn(support3, 16, ratio=0.25)
            ids5, support5 = sampling_knn(support4, 16, ratio=0.25)
            return None, [ids1, ids2, ids3, ids4, ids5], [support1, support2, support3, support4, support5]


    def forward_with_features(self, x, pos, support_points=None, indices=None):

        if (support_points is None) or (indices is None):
            _, indices, support_points = self.compute_indices(pos)

        if self.segmentation:

            ids0, ids1, ids2, ids3, ids4, ids5, ids6, ids5d, ids4d, ids3d, ids2d, ids1d, ids0d = indices
            support0, support1, support2, support3, support4, support5, support6 = support_points

            x0 = self.activation(self.bn0(self.cv0(x, pos, support0, ids0)))
            x1 = self.activation(self.bn1(self.cv1(x0, support0, support1, ids1)))
            x2 = self.activation(self.bn2(self.cv2(x1, support1, support2, ids2)))
            x3 = self.activation(self.bn3(self.cv3(x2, support2, support3, ids3)))
            x4 = self.activation(self.bn4(self.cv4(x3, support3, support4, ids4)))
            x5 = self.activation(self.bn5(self.cv5(x4, support4, support5, ids5)))
            x6 = self.activation(self.bn6(self.cv6(x5, support5, support6, ids6)))
            x = self.activation(self.bn5d(self.cv5d(x6, support6, support5, ids5d)))
            x = torch.cat([x, x5], dim=1)
            x = self.activation(self.bn4d(self.cv4d(x, support5, support4, ids4d)))
            x = torch.cat([x, x4], dim=1)
            x = self.activation(self.bn3d(self.cv3d(x, support4, support3, ids3d)))
            x = torch.cat([x, x3], dim=1)
            x = self.activation(self.bn2d(self.cv2d(x, support3, support2, ids2d)))
            x = torch.cat([x, x2], dim=1)
            x = self.activation(self.bn1d(self.cv1d(x, support2, support1, ids1d)))
            x = torch.cat([x, x1], dim=1)
            x = self.activation(self.bn0d(self.cv0d(x, support1, support0, ids0d)))
            x = torch.cat([x, x0], dim=1)
            x = self.dropout(x)
            x = self.fcout(x)

        else:

            ids1, ids2, ids3, ids4, ids5 = indices
            support1, support2, support3, support4, support5 = support_points

            x = self.activation(self.bn1(self.cv1(x, pos, support1, ids1)))
            x = self.activation(self.bn2(self.cv2(x, support1, support2, ids2)))
            x = self.activation(self.bn3(self.cv3(x, support2, support3, ids3)))
            x = self.activation(self.bn4(self.cv4(x, support3, support4, ids4)))
            x = self.activation(self.bn5(self.cv5(x, support4, support5, ids5)))
            x = x.mean(dim=2)
            x = self.dropout(x)
            x = self.fcout(x)

        return x


    def forward(self, x, pos, support_points=None, indices=None):
        if x is None:
            return self.compute_indices(pos)
        else:
            return self.forward_with_features(x, pos, support_points, indices)
