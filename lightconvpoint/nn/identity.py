import torch.nn as nn


class Identity(nn.Module):
    """Indentity module for LCP."""

    def __init__(self):
        super().__init__()

    def forward(self, input, points, support_points=None, indices=None):
        return input, support_points, indices
