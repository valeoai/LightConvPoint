import torch
import torch.nn as nn


class MaxPool(nn.Module):
    def __init__(self, search=None):
        """
        ConvPt: network for computing the convolution
        SearchPt: spatial knn or radius search function
        """
        super().__init__()
        self.search = search

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

    def forward(self, input, points, support_points=None, indices=None):
        """
        Forward function of the layer
        """

        if indices is None and self.search is None:
            raise Exception(
                "MaxPool - valid search method is required if indices are not provided"
            )

        if indices is None:
            # search the support points and the neighborhoods
            indices, support_points = self.search(points, support_points)

        if input is None:
            # inpuy is None: do not compute features
            return None, support_points, indices
        else:
            # compute the features
            indices = indices.clone()

            # get the features coordinates associated with the indices
            features = self.batched_index_select(
                input, dim=2, index=indices
            ).contiguous()

            features = features.max(dim=3)[0]

            return features, support_points, indices
