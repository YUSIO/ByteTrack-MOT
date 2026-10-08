"""YOLOX with a position-prior map injected into the stride-8 neck feature (Exp064).

Training targets may carry prior rows next to the object rows: [k, cx, cy, w, h], where the class column holds
k >= 1, the number of frames since the member was last observed (object rows have class 0). They went through the same augmentation as the objects, are rendered here into a
two-channel map and removed before the loss. At inference the priors are passed explicitly.

The injection is residual and its last convolution starts at zero, so a model initialised from a plain YOLOX
checkpoint reproduces that checkpoint's outputs exactly before training.
"""
import torch
import torch.nn as nn

from .network_blocks import BaseConv
from .yolox import YOLOX

STRIDE = 8


def render_prior(priors, hw, dtype, device):
    """priors: per image a tensor [n, 4] of (cx, cy, size, k) in network-input pixels -> [B, 2, h, w].

    Channel 0 is the max over priors of a Gaussian whose width grows with the gap; channel 1 weights it by k / 30.
    """
    h, w = hw
    out = torch.zeros(len(priors), 2, h, w, dtype=torch.float32, device=device)
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * STRIDE
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * STRIDE
    for b, p in enumerate(priors):
        if p is None or len(p) == 0:
            continue
        p = p.to(device=device, dtype=torch.float32)
        k = p[:, 3].clamp(1, 30)
        sigma = p[:, 2].clamp(min=float(STRIDE)) * (0.5 + k / 20)
        d2 = (xs[None, None, :] - p[:, 0, None, None]) ** 2 + (ys[None, :, None] - p[:, 1, None, None]) ** 2
        g = torch.exp(-d2 / (2 * sigma[:, None, None] ** 2))
        out[b, 0] = g.max(0).values
        out[b, 1] = (g * (k / 30)[:, None, None]).max(0).values
    return out.to(dtype)


class PriorYOLOX(YOLOX):
    def __init__(self, backbone=None, head=None, channels=128, use_prior=True):
        super().__init__(backbone, head)
        self.use_prior = use_prior
        self.inject = nn.Sequential(BaseConv(channels + 2, channels, 3, 1, act="silu"), nn.Conv2d(channels, channels, 1))
        nn.init.zeros_(self.inject[1].weight)
        nn.init.zeros_(self.inject[1].bias)

    @staticmethod
    def split_targets(targets):
        """Separate object rows from prior rows; object rows are packed to the front as the loss expects."""
        objects, priors = torch.zeros_like(targets), []
        for b in range(targets.shape[0]):
            t = targets[b]
            valid = t[:, 1:5].sum(1) > 0
            is_prior = valid & (t[:, 0] > 0.5)
            keep = t[valid & ~is_prior]
            objects[b, : len(keep)] = keep
            p = t[is_prior]
            priors.append(torch.stack((p[:, 1], p[:, 2], torch.sqrt(p[:, 3] * p[:, 4]), p[:, 0]), 1))
        return objects, priors

    def forward(self, x, targets=None, prior=None):
        if self.training:
            assert targets is not None
            targets, prior = self.split_targets(targets)
        fpn_outs = list(self.backbone(x))
        f = fpn_outs[0]
        if not self.use_prior or prior is None:
            prior = [None] * x.shape[0]
        heat = render_prior(prior, f.shape[2:], f.dtype, f.device)
        fpn_outs[0] = f + self.inject(torch.cat((f, heat), 1))

        if self.training:
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(fpn_outs, targets, x)
            return {"total_loss": loss, "iou_loss": iou_loss, "l1_loss": l1_loss, "conf_loss": conf_loss,
                    "cls_loss": cls_loss, "num_fg": num_fg}
        return self.head(fpn_outs)
