import torch.nn as nn


class Identity(nn.Module):
    """Indentity module compatible with LightConvPoint.

    # Forward arguments
        input: 3-D torch tensor.
            Input feature tensor. Dimensions are (B, I, N) with B the batch size,
            I the number of input channels and N the number of input points.
        points: 3-D torch tensor.
            The input points. Dimensions are (B, D, N) with B the batch size, D the
            dimension of the spatial space and N the number of input points.
        support_points: (optional) 3-D torch tensor.
            The support points to project features on. If not provided, use the `search`
            object of the layer to compute them.
            Dimensions are (B, D, N) with B the batch size, D the dimenstion of the
            spatial space and N the number of input points.
        indices: (optional) 3-D torch tensor.
            The indices of the neighboring points with respect to the support points.
            If not provided, use the `search` object of the layer to compute them.

    # Returns
        (input, support_points, indices)
    """

    def __init__(self):
        super().__init__()

    def forward(self, input, points, support_points=None, indices=None):
        return input, support_points, indices
