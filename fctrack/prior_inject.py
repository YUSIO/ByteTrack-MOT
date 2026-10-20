"""Exp120: inject the slot prior heatmap (fctrack.slot_prior) into a detector's FPN features, before its head.

For FPN level k with features F_k: F'_k = F_k + M_k * phi_k([F_k, E_k]), where E = enc(prior) is an embedding of the
heatmap at stride 8 (max-pooled to the level's resolution), phi_k is a small convolution on the features together with
the embedding (so the injection depends on what the image shows at the slot), and M_k = max-pooled max of the prior
channels: the features change only where a slot is, never on the rest of the image. The last convolution of phi_k starts
at zero, so the untrained module is the original detector. Only the channel counts of the FPN levels are needed.

Exp122: with `bound` b the residual is b * tanh(phi_k / b), so no feature moves by more than b (Exp120: the unbounded residual
reached about twice the feature range at stride 32, the regression's log-size output overflowed and the gradients became NaN).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorInjection(nn.Module):
    def __init__(self, channels, prior_channels=2, emb=32, hidden=64, bound=None):
        super().__init__()
        self.bound = bound
        self.enc = nn.Sequential(nn.Conv2d(prior_channels, emb, 3, padding=1), nn.SiLU(), nn.Conv2d(emb, emb, 3, padding=1), nn.SiLU())
        self.fuse = nn.ModuleList(nn.Sequential(nn.Conv2d(c + emb, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(),
                                                nn.Conv2d(hidden, c, 1)) for c in channels)
        for f in self.fuse:
            nn.init.zeros_(f[-1].weight)
            nn.init.zeros_(f[-1].bias)

    def forward(self, feats, prior):
        prior = prior.to(feats[0].dtype)
        e = self.enc(prior)
        m = prior.amax(1, keepdim=True)
        out = []
        for k, f in enumerate(feats):
            hw = f.shape[-2:]
            ek = e if e.shape[-2:] == hw else F.adaptive_max_pool2d(e, hw)
            mk = m if m.shape[-2:] == hw else F.adaptive_max_pool2d(m, hw)
            r = self.fuse[k](torch.cat([f, ek], 1))
            if self.bound:
                r = self.bound * torch.tanh(r / self.bound)
            out.append(f + mk * r)
        return out


class PriorYOLOX(nn.Module):
    """A YOLOX model with the injection between its backbone (YOLOPAFPN) and its head; inj None = the plain detector."""

    def __init__(self, yolox, inj=None):
        super().__init__()
        self.det, self.inj = yolox, inj

    def features(self, x, prior=None):
        f = list(self.det.backbone(x))
        return f if self.inj is None or prior is None else self.inj(f, prior)

    def forward(self, x, prior=None):
        return self.det.head(self.features(x, prior))

    def losses(self, x, targets, prior=None, both=True):
        """Head losses with the prior (when given) and without it; returns a dict of the two loss tuples."""
        f = list(self.det.backbone(x))
        out = {"plain": self.det.head(f, targets, x)} if both or prior is None or self.inj is None else {}
        if prior is not None and self.inj is not None:
            out["prior"] = self.det.head(self.inj(f, prior), targets, x)
        return out
