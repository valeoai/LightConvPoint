import torch
import torch.nn as nn
import lightconvpoint.knn as nearest_neighbors


class UpSampleNearest(nn.Module):
    def __init__(self):
        """
        ConvPt: network for computing the convolution
        SearchPt: spatial knn or radius search function
        """
        super().__init__()

    def batched_index_select(self, input, dim, index):
        """
        Slicing of input with respect to the index tensor
        """
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

    def forward(self, input, points, support_points, indices=None):
        """
        Forward function of the layer
        """

        # support points are known, only compute the knn
        if indices is None:
            indices = nearest_neighbors.knn(
                points.cpu().detach(), support_points.cpu().detach(), 1
            )
            if points.is_cuda:
                indices = indices.cuda()

        if input is None:
            # inpuy is None: do not compute features
            return None, support_points, indices
        else:
            # compute the features
            indices = indices.clone()

            # get the features and point coordinates associated with the indices
            features = self.batched_index_select(
                input, dim=2, index=indices
            ).contiguous()

            return features.squeeze(3), support_points, indices
