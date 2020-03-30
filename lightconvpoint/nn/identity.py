import torch
import torch.nn as nn
import numpy as np


class Identity(nn.Module):
    """Indentity module for LCP."""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input, points, support_points=None, indices=None):
        return input, support_points, indices
