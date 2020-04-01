import torch.nn as nn


class Identity(nn.Module):
    """Indentity module compatible with LightConvPoint.
    
    # Returns
        (input, support_points, indices)
    """

    def __init__(self):
        super().__init__()

    def forward(self, input, points, support_points=None, indices=None):
        return input, support_points, indices
