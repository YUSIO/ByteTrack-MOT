"""PPTracker Eq. (1), with an explicit, non-author token-to-image bridge.

The paper leaves N x D -> 3 x H x W undefined. We reuse the attention
weights to scatter prompt values onto the patch grid, then unpatchify.
No heatmap-only substitute, temporal image features, or tracker edits.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class PriorPrompt(nn.Module):
    def __init__(self, dim=64, patch=16, bins=1024, max_ids=4096):
        super().__init__()
        self.dim, self.patch, self.bins, self.max_ids = dim, patch, bins, max_ids
        self.coordinate = nn.ModuleList([nn.Embedding(bins, dim) for _ in range(4)])
        self.identity = nn.Embedding(max_ids, dim)
        self.patch_embed = nn.Conv2d(3, dim, patch, stride=patch)
        self.key, self.value = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.to_patch = nn.Linear(dim, 3 * patch * patch, bias=False)
        # Exact baseline at initialization; not specified by the paper.
        nn.init.zeros_(self.to_patch.weight)

    def forward(self, image, boxes, ids):
        """One frame; boxes are xywh in detector-input pixels, IDs are persistent.

        Variable numbers of targets are processed separately, without padded
        attention tokens. Coordinates are clipped only for token quantization;
        detection boxes and tracker observations are never clipped here.
        """
        if image.shape[0] != 1 or boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError('Expected image [1,3,H,W] and boxes [N,4]')
        if ids.shape != (len(boxes),) or len(torch.unique(ids)) != len(ids):
            raise ValueError('Prompt IDs must be unique, one per box')
        if not len(boxes):
            return image
        if ids.min() < 0 or ids.max() >= self.max_ids:
            raise ValueError('Prompt ID outside vocabulary; never wrap IDs modulo vocabulary')
        if not torch.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any():
            raise ValueError('Invalid prompt boxes')
        h, w = image.shape[-2:]
        norm = boxes / boxes.new_tensor([w, h, w, h])
        index = (norm.clamp(0, 1) * (self.bins - 1)).round().long()
        q = self.identity(ids.long())
        for i, embedding in enumerate(self.coordinate):
            q = q + embedding(index[:, i])
        padded = F.pad(image, (0, (-w) % self.patch, 0, (-h) % self.patch))
        patches = self.patch_embed(padded).flatten(2).transpose(1, 2)[0]
        attention = (q @ self.key(patches).T / math.sqrt(self.dim)).softmax(-1)
        prompt = attention @ self.value(patches)  # PPTracker Eq. (1), N x D
        grid = attention.T @ prompt / len(boxes)  # declared shape-completion choice
        delta = F.fold(self.to_patch(grid).T.unsqueeze(0), padded.shape[-2:],
                       kernel_size=self.patch, stride=self.patch)
        return image + delta[:, :, :h, :w]  # Eq. (2), before backbone
