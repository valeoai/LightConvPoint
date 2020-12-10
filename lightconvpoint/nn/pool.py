import torch
import torch.nn as nn


def max_pool(input, indices):

    dim=2
    index_shape = indices.shape
    views = [input.shape[0]] + [
        1 if i != dim else -1 for i in range(1, len(input.shape))
    ]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    indices = indices.view(views).expand(expanse)

    features = torch.gather(input, dim, indices).view(input.size(0), -1, index_shape[1], index_shape[2])
    features = features.max(dim=3)[0]
    return features

class MaxPool(nn.Module):

    def __init__(self):
        super().__init__()

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

    def forward(self, input, points, support_points, indices):
        """Forward function of the layer."""
        features = self.batched_index_select(input, dim=2, index=indices).contiguous()
        features = features.max(dim=3)[0]
        return features