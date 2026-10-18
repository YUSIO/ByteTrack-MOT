"""Temporal feature fusion attached between a frozen detector's FPN and its frozen head (Exp091, Idea 0035 base B).

For each FPN level: F'_t = F_t + sigmoid(W_g [F_t, A]) * W_v A, with A the previous frame's feature of the same level,
either not aligned (`zero`) or sampled backwards through a camera-motion affine (`gmc`). W_v starts at zero, so the
untrained module is the original detector. Only the channel counts of the levels are needed: the same module attaches
to any detector that exposes FPN levels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalFusion(nn.Module):
    def __init__(self, channels, align="zero"):
        super().__init__()
        assert align in ("zero", "gmc")
        self.align = align
        self.gate = nn.ModuleList(nn.Conv2d(2 * c, c, 1) for c in channels)
        self.value = nn.ModuleList(nn.Conv2d(c, c, 1) for c in channels)
        for v in self.value:
            nn.init.zeros_(v.weight)
            nn.init.zeros_(v.bias)

    @staticmethod
    def warp(prev, m_inv, stride):
        """Sample `prev` (B, C, H, W) at M^-1 x for every output location x (input-pixel coordinates).
        m_inv: (B, 3, 3) maps input coordinates of frame t to frame t-1."""
        b, _, h, w = prev.shape
        ys, xs = torch.meshgrid(torch.arange(h, device=prev.device), torch.arange(w, device=prev.device), indexing="ij")
        pts = torch.stack([(xs + 0.5) * stride, (ys + 0.5) * stride, torch.ones_like(xs, dtype=torch.float32)], 0).float().view(3, -1)
        src = (m_inv @ pts)[:, :2]  # (B, 2, HW)
        gx = 2 * src[:, 0] / (stride * w) - 1
        gy = 2 * src[:, 1] / (stride * h) - 1
        grid = torch.stack([gx, gy], -1).view(b, h, w, 2)
        return F.grid_sample(prev, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def forward(self, cur, prev, m_inv=None, input_size=None):
        out = []
        for lvl, (c, p) in enumerate(zip(cur, prev)):
            if self.align == "gmc":
                p = self.warp(p, m_inv, input_size[0] // c.shape[-2])
            g = torch.sigmoid(self.gate[lvl](torch.cat([c, p], 1)))
            out.append(c + g * self.value[lvl](p))
        return out

    def gate_means(self, cur, prev, m_inv=None, input_size=None):
        with torch.no_grad():
            means = []
            for lvl, (c, p) in enumerate(zip(cur, prev)):
                if self.align == "gmc":
                    p = self.warp(p, m_inv, input_size[0] // c.shape[-2])
                means.append(float(torch.sigmoid(self.gate[lvl](torch.cat([c, p], 1))).mean()))
            return means
