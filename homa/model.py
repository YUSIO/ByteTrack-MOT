"""MPA-Net structure from Chu et al. (2025), Eqs. 1-7, 22-23.

Not an author implementation. Projection sizes, residual normalization and the
extra fused-logit supervision are explicit independent implementation choices.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50, ResNet50_Weights


class PoseBlock(nn.Module):
    def __init__(self, dim, latent=256, heads=8, hidden=512):
        super().__init__()
        self.patch_dim = dim
        self.hist_proj = nn.Linear(dim, latent)
        self.curr_proj = nn.Linear(dim, latent)
        self.sa = nn.MultiheadAttention(latent, heads, batch_first=True)
        self.ca = nn.MultiheadAttention(latent, heads, batch_first=True)
        self.hnorm = nn.LayerNorm(latent)
        self.cnorm = nn.LayerNorm(latent)
        self.hmlp = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, dim), nn.ReLU())
        self.cmlp = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, dim), nn.ReLU())
        self.hkey = nn.Linear(dim, latent)

    def forward(self, current, history):
        h = self.hist_proj(history).unsqueeze(0)
        h = self.hnorm(h + self.sa(h, h, h, need_weights=False)[0])
        enhanced = self.hmlp(h).squeeze(0)
        q = self.curr_proj(current).unsqueeze(0)
        k = self.hkey(enhanced).unsqueeze(0)
        c = self.cnorm(q + self.ca(q, k, k, need_weights=False)[0])
        enhanced_current = self.cmlp(c).squeeze(0)
        # sqrt(D) scaling is fixed, documented, never selected on official test.
        return enhanced_current.float() @ enhanced.float().T / math.sqrt(self.patch_dim)


class MPANet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        r = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)
        self.reduce = nn.Conv2d(1024, 256, 1)
        self.patch = nn.ModuleList([nn.Conv2d(256, 512, 2, stride=2) for _ in range(3)])
        self.pose = nn.ModuleList([PoseBlock(d) for d in (2048, 4096, 2048)])
        self.fuse = nn.Conv2d(3, 1, 1)
        nn.init.constant_(self.fuse.weight, 1 / 3)
        nn.init.constant_(self.fuse.bias, 0.1)

    def train(self, mode=True):
        super().train(mode)
        # Tiny, variable numbers of instances per microbatch: fixed pretrained BN.
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def encode(self, crops):
        if crops.shape[0] == 0:
            return crops.new_empty((0, 8192))
        f = self.reduce(self.backbone(crops))
        assert f.shape[-2:] == (8, 8), f.shape
        blocks = f.split((2, 4, 2), dim=3)
        return torch.cat([p(b).flatten(1) for p, b in zip(self.patch, blocks)], dim=1)

    def logits(self, current, history):
        cs, hs = current.split((2048, 4096, 2048), 1), history.split((2048, 4096, 2048), 1)
        parts = torch.stack([p(c, h) for p, c, h in zip(self.pose, cs, hs)], dim=0)
        fused = F.relu(self.fuse(parts.unsqueeze(0)).squeeze(0).squeeze(0))
        return parts, fused

    def forward(self, crops, n_current):
        e = self.encode(crops)
        return self.logits(e[:n_current], e[n_current:])


def association_loss(parts, fused, current_ids, history_ids, history_frames):
    """Per-history-frame categorical association; absent identities are masked.

Three part losses implement the Eq. 22-23 supervision; an explicit additional
fused loss trains Eq. 5's conv, whose training path is unspecified in the paper.
"""
    losses, fused_losses, n = [], [], 0
    for frame in torch.unique(history_frames):
        cols = history_frames == frame
        labels = current_ids[:, None] == history_ids[cols][None, :]
        present = labels.any(1)
        if not present.any():
            continue
        target = labels[present].float().argmax(1)
        losses.extend(F.cross_entropy(part[present][:, cols].float(), target) for part in parts)
        fused_losses.append(F.cross_entropy(fused[present][:, cols].float(), target))
        n += int(present.sum())
    if not losses:
        return fused.sum() * 0, 0
    return torch.stack(losses).mean() * 3 + torch.stack(fused_losses).mean(), n


def tracklet_similarity(logits, history_frames, history_owners, n_tracks):
    """Eq. 6 per-frame softmax then Eq. 7 sum; owners are ONLINE track IDs."""
    result = logits.new_zeros((logits.shape[0], n_tracks), dtype=torch.float32)
    for frame in torch.unique(history_frames):
        cols = history_frames == frame
        probs = logits[:, cols].float().softmax(1)
        owners = history_owners[cols]
        result.scatter_add_(1, owners[None].expand(logits.shape[0], -1), probs)
    return result
