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
    def __init__(self, dim=64, patch=16, bins=1024, max_ids=4096, position_encoding=False, content_score="scaled_dot",
                 backprojection="mean", projection_init_gain=0., spatial_score="learned_fourier",
                 value_content="raw"):
        super().__init__()
        if value_content not in ("raw", "patch_detail"):
            raise ValueError("Unknown value content")
        self.value_content = value_content
        if content_score not in ("scaled_dot", "cosine"):
            raise ValueError("Unknown content score")
        if backprojection not in ("mean", "patch_mean_compensated", "peak_coverage"):
            raise ValueError("Unknown backprojection")
        if projection_init_gain < 0 or not math.isfinite(projection_init_gain):
            raise ValueError("Invalid projection initialization gain")
        if spatial_score not in ("learned_fourier", "box_gaussian"):
            raise ValueError("Unknown spatial score")
        if spatial_score == "box_gaussian" and content_score != "cosine":
            raise ValueError("Box geometry requires bounded cosine content")
        self.spatial_score = spatial_score
        self.backprojection = backprojection
        self.content_score = content_score
        self.position_encoding = position_encoding
        if position_encoding and dim % 4:
            raise ValueError("2D Fourier position encoding requires dim divisible by 4")
        self.dim, self.patch, self.bins, self.max_ids = dim, patch, bins, max_ids
        self.coordinate = nn.ModuleList([nn.Embedding(bins, dim) for _ in range(4)])
        self.identity = nn.Embedding(max_ids, dim)
        self.patch_embed = nn.Conv2d(3, dim, patch, stride=patch)
        self.key, self.value = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.to_patch = nn.Linear(dim, 3 * patch * patch, bias=False)
        # Exact baseline at initialization; not specified by the paper.
        if projection_init_gain == 0:
            nn.init.zeros_(self.to_patch.weight)
        else:
            # Small nonzero fan-in initialization lets detection loss reach Q/K/V.
            nn.init.normal_(self.to_patch.weight, std=projection_init_gain / math.sqrt(dim))

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
        padded = F.pad(image, (0, (-w) % self.patch, 0, (-h) % self.patch))
        patches = self.patch_embed(padded).flatten(2).transpose(1, 2)[0]
        q = self.encode_query(boxes, ids, h, w)
        attention = self.attention(q, patches, h, w, boxes)
        prompt = attention @ self.patch_values(padded, patches)
        grid = self.backproject(attention, prompt)
        delta = F.fold(self.to_patch(grid).T.unsqueeze(0), padded.shape[-2:],
                       kernel_size=self.patch, stride=self.patch)
        return image + delta[:, :, :h, :w]  # Eq. (2), before backbone

    def patch_values(self, padded, patches):
        """Content-only ablation; Q/K still receive the original image tokens.

        Subtract the response to each patch's per-channel mean. Both affine
        constants cancel as well: V(X)-V(mean(X)) = Wv We (X-mean(X)).
        No RMS normalization, gain, new parameters, or change of attention.
        Means include the existing zero padding on partial boundary patches.
        """
        values = self.value(patches)
        if self.value_content == "raw":
            return values
        means = F.avg_pool2d(padded, self.patch, stride=self.patch)
        mean_image = means.repeat_interleave(self.patch, -2).repeat_interleave(self.patch, -1)
        mean_patches = self.patch_embed(mean_image).flatten(2).transpose(1, 2)[0]
        return values - self.value(mean_patches)

    def backproject(self, attention, prompt):
        """Declared migration bridge; compensate uniform 1/M spatial dilution.

        Keep spatial weights, including the one-object case. Dividing each patch
        by its attention mass would cancel all spatial dependence when N=1.
        Sharp attention can still amplify local values; real-loss gates audit it.
        """
        if self.backprojection == "peak_coverage":
            # Each object writes at most unit weight. Blend overlapping writes;
            # preserve sub-unit support when N=1 instead of cancelling it.
            support = attention / attention.amax(-1, keepdim=True)
            return (support.T @ prompt) / support.sum(0)[:, None].clamp_min(1.)
        grid = attention.T @ prompt / len(prompt)
        if self.backprojection == "patch_mean_compensated":
            grid = grid * attention.shape[1]
        return grid

    def encode_query(self, boxes, ids, h, w):
        norm = boxes / boxes.new_tensor([w, h, w, h])
        index = (norm.clamp(0, 1) * (self.bins - 1)).round().long()
        q = self.identity(ids.long())
        for i, embedding in enumerate(self.coordinate):
            q = q + embedding(index[:, i])
        return q

    def patch_positions(self, h, w, reference):
        # Positions describe patch centers in the unpadded detector-input frame.
        rows, cols = math.ceil(h / self.patch), math.ceil(w / self.patch)
        y, x = torch.meshgrid(torch.arange(rows, device=reference.device),
                              torch.arange(cols, device=reference.device), indexing='ij')
        xy = torch.stack(((x + .5) * self.patch / w,
                          (y + .5) * self.patch / h), dim=-1).reshape(-1, 2).to(reference)
        frequency = torch.arange(1, self.dim // 4 + 1, device=reference.device).to(reference)
        phase = 2 * math.pi * xy[..., None] * frequency
        return torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(1)

    def content_logits(self, q, keys):
        if self.content_score == "cosine":
            # Fixed unit-temperature cosine: bounded [-1, 1], no tuned gain.
            return F.normalize(q, dim=-1, eps=1e-6) @ F.normalize(keys, dim=-1, eps=1e-6).T
        return q @ keys.T / math.sqrt(self.dim)

    def geometry_logits(self, boxes, h, w, reference):
        """Continuous prior geometry; no current GT or learned bin-to-pixel map.

        One patch is the minimum sigma; larger boxes use half their width/height.
        No wraparound in the image plane. Previous-box error remains a limitation.
        """
        rows, cols = math.ceil(h / self.patch), math.ceil(w / self.patch)
        y, x = torch.meshgrid(torch.arange(rows, device=reference.device),
                              torch.arange(cols, device=reference.device), indexing="ij")
        centers = torch.stack((x + .5, y + .5), -1).reshape(-1, 2).to(reference) * self.patch
        center = boxes[:, :2] + boxes[:, 2:] / 2
        sigma = (boxes[:, 2:] / 2).clamp_min(self.patch)
        return -.5 * (((centers[None] - center[:, None]) / sigma[:, None]) ** 2).sum(-1)

    def attention_logits(self, q, patches, h, w, boxes=None):
        keys = self.key(patches)
        if self.spatial_score == "box_gaussian":
            logits = self.content_logits(q, keys)
            if self.position_encoding:
                if boxes is None or len(boxes) != len(q):
                    raise ValueError("One historical box per query required")
                logits = logits + self.geometry_logits(boxes, h, w, keys)
            return logits
        if self.content_score == "scaled_dot":
            # Preserve the exact previous implementation for old configs.
            if self.position_encoding:
                keys = keys + self.patch_positions(h, w, keys)
            return q @ keys.T / math.sqrt(self.dim)
        logits = self.content_logits(q, keys)
        if self.position_encoding:
            logits = logits + q @ self.patch_positions(h, w, keys).T / math.sqrt(self.dim)
        return logits

    def attention(self, q, patches, h, w, boxes=None):
        return self.attention_logits(q, patches, h, w, boxes).softmax(-1)
