"""MIT TraDeS DCN_TraDeS parameterization, torchvision operator substitution.

Original custom extension parity remains unverified; CPU sampling conventions
and gradients are tested. No silent fallback to ordinary convolution.
"""
import math
import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d


class DCN_TraDeS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation=1, deformable_groups=1):
        super().__init__()
        if deformable_groups != 1:
            raise ValueError('Only the original one-group TraDeS operator is supported')
        self.kernel_size = _pair(kernel_size)
        self.stride, self.padding, self.dilation = map(_pair, (stride, padding, dilation))
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.uniform_(self.weight, -1/math.sqrt(in_channels*math.prod(self.kernel_size)),
                         1/math.sqrt(in_channels*math.prod(self.kernel_size)))

    def forward(self, x, offset, mask):
        return deform_conv2d(x, offset, self.weight, self.bias, self.stride,
                             self.padding, self.dilation, mask)
