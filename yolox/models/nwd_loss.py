"""Pure normalized Wasserstein box loss in augmented input-pixel cxcywh units."""
import math
import torch
from torch import nn


class NWDloss(nn.Module):
    def __init__(self, constant=12.8, reduction="none"):
        super().__init__()
        if not math.isfinite(constant) or constant <= 0:
            raise ValueError("constant must be finite and positive")
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("unsupported reduction")
        self.constant = float(constant)
        self.reduction = reduction

    def forward(self, pred, target):
        if pred.shape != target.shape or pred.shape[-1] != 4:
            raise ValueError("pred and target must have matching (..., 4) shape")
        # Explicit FP32 before subtraction; norm has zero subgradient at zero.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            delta = pred.float().reshape(-1, 4) - target.float().reshape(-1, 4)
            scaled = torch.cat((delta[:, :2], delta[:, 2:] * 0.5), dim=1)
            distance = torch.linalg.vector_norm(scaled, ord=2, dim=1)
            loss = -torch.expm1(-distance / self.constant)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
