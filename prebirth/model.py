"""A small dense detector with controlled single-frame/temporal readouts.

Input frames have shape [B, 3, 3, H, W], in the order current, t-3, t-6.
They are RGB floats in [0, 1]; no ImageNet normalization or download is needed.
Outputs have spatial size ceil(H/4), ceil(W/4). ``offset`` is an unconstrained
grid-cell residual, and ``log_wh`` is log(width, height) in input-image pixels.

``aligned`` considers nine constant-velocity paths within this *finite* input
window. It neither carries accumulated evidence between calls nor consumes
GT paths, detector boxes, identities, or future images. Softmax is a feature
selection operation, not a calibrated target-existence posterior. The original
historical positions remain available as context; the zero-velocity path is
always valid, including for hovering objects and image boundaries.
"""

import math
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    # At least two channels per group also permits a 1x1 feature map at B=1.
    return next(group for group in (8, 4, 2, 1) if channels % group == 0 and channels // group >= 2)


class _ConvNorm(nn.Sequential):
    def __init__(self, incoming: int, outgoing: int, stride: int = 1, kernel: int = 3):
        super().__init__(
            nn.Conv2d(incoming, outgoing, kernel, stride, kernel // 2, bias=False),
            nn.GroupNorm(_groups(outgoing), outgoing),
            nn.SiLU(inplace=True),
        )


class _Residual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            _ConvNorm(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )

    def forward(self, features: Tensor) -> Tensor:
        return F.silu(features + self.body(features))


def _shift_without_wrap(features: Tensor, dx: int, dy: int) -> Tuple[Tensor, Tensor]:
    """Move source features by (+dx,+dy), returning a broadcastable valid mask.

    Thus output[y,x] = source[y-dy,x-dx]. Padding never reads the opposite
    image edge. The mask has shape [1,1,H,W] and describes actual source support.
    """
    height, width = features.shape[-2:]
    if abs(dx) >= width or abs(dy) >= height:
        return features * 0.0, torch.zeros(
            (1, 1, height, width), dtype=torch.bool, device=features.device
        )
    source = features[
        ..., max(-dy, 0):min(height - dy, height),
        max(-dx, 0):min(width - dx, width)
    ]
    padding = (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0))
    shifted = F.pad(source, padding)
    valid = F.pad(
        torch.ones((1, 1, *source.shape[-2:]), dtype=torch.bool, device=features.device),
        padding,
        value=False,
    )
    return shifted, valid


class TemporalDetector(nn.Module):
    """Same trainable architecture for ``single``, ``fusion``, and ``aligned``.

    Single-frame mode encodes only the current frame and repeats its features
    into the temporal input slots. Fusion reads the three unaligned frames.
    Aligned additionally supplies the coherent-path-weighted historical features.
    All modes use identical encoder, fusion block, and output heads. Compute
    differs and should be measured rather than inferred from parameter count.
    """

    stride = 4
    input_frames = 3
    MODES = ("single", "fusion", "aligned")

    def __init__(
        self,
        mode: str = "aligned",
        channels: int = 48,
        path_step: int = 2,
        temperature: float = 0.2,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError("mode must be one of {}".format(self.MODES))
        if not isinstance(channels, int) or channels < 8:
            raise ValueError("channels must be an integer >= 8")
        if not isinstance(path_step, int) or path_step < 1:
            raise ValueError("path_step must be a positive integer in feature cells")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.mode = mode
        self.channels = channels
        self.path_step = path_step
        self.temperature = float(temperature)
        self.path_shifts = tuple(
            (dx, dy)
            for dy in (-path_step, 0, path_step)
            for dx in (-path_step, 0, path_step)
        )
        middle = max(8, channels // 2)
        self.encoder = nn.Sequential(
            _ConvNorm(3, middle, stride=2),
            _Residual(middle),
            _ConvNorm(middle, channels, stride=2),
            _Residual(channels),
            _Residual(channels),
        )
        # Raw-position history is supplied separately from aligned history.
        self.fusion = nn.Sequential(
            _ConvNorm(5 * channels, channels, kernel=1),
            _Residual(channels),
        )
        self.heatmap = self._head(channels, 1)
        self.offset = self._head(channels, 2)
        self.log_wh = self._head(channels, 2)
        self._initialize()

    @staticmethod
    def _head(channels: int, outputs: int) -> nn.Sequential:
        return nn.Sequential(_ConvNorm(channels, channels), nn.Conv2d(channels, outputs, 1))

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for head in (self.heatmap, self.offset, self.log_wh):
            nn.init.normal_(head[-1].weight, std=0.001)
        nn.init.constant_(self.heatmap[-1].bias, math.log(0.01 / 0.99))
        nn.init.constant_(self.offset[-1].bias, 0.5)
        nn.init.constant_(self.log_wh[-1].bias, math.log(24.0))

    @property
    def model_config(self) -> Dict[str, object]:
        """Save with state_dict, then restore with TemporalDetector(**config)."""
        return {
            "mode": self.mode,
            "channels": self.channels,
            "path_step": self.path_step,
            "temperature": self.temperature,
        }

    def _path_weights(self, current: Tensor, previous: Tensor, older: Tensor) -> Tensor:
        # Float32 matching remains finite during mixed-precision training. The
        # same displacement hypothesis is used at both time offsets.
        current = F.normalize(current.float(), dim=1, eps=1e-6)
        previous = F.normalize(previous.float(), dim=1, eps=1e-6)
        older = F.normalize(older.float(), dim=1, eps=1e-6)
        logits = []
        for dx, dy in self.path_shifts:
            first, first_valid = _shift_without_wrap(previous, dx, dy)
            second, second_valid = _shift_without_wrap(older, 2 * dx, 2 * dy)
            score = 0.5 * (current * (first + second)).sum(dim=1, keepdim=True)
            score = score / self.temperature
            logits.append(score.masked_fill(~(first_valid & second_valid), float("-inf")))
        # The zero path is valid everywhere, so every pixel has a finite entry.
        return torch.softmax(torch.cat(logits, dim=1), dim=1)

    def _aligned_history(self, current: Tensor, previous: Tensor, older: Tensor) -> Tuple[Tensor, Tensor]:
        weights = self._path_weights(current, previous, older).to(previous.dtype)
        aligned_previous = torch.zeros_like(previous)
        aligned_older = torch.zeros_like(older)
        for index, (dx, dy) in enumerate(self.path_shifts):
            first, _ = _shift_without_wrap(previous, dx, dy)
            second, _ = _shift_without_wrap(older, 2 * dx, 2 * dy)
            weight = weights[:, index:index + 1]
            aligned_previous = aligned_previous + weight * first
            aligned_older = aligned_older + weight * second
        return aligned_previous, aligned_older

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        if images.ndim != 5 or images.shape[1:3] != (self.input_frames, 3):
            raise ValueError("images must have shape [B,3,3,H,W], ordered current,t-3,t-6")
        if not images.is_floating_point():
            raise TypeError("images must be floating-point RGB values in [0,1]")
        if images.shape[0] < 1 or min(images.shape[-2:]) < 4:
            raise ValueError("images require a nonempty batch and spatial dimensions >= 4")
        if self.mode == "single":
            current = self.encoder(images[:, 0])
            previous = older = aligned_previous = aligned_older = current
        else:
            batch, frames, colors, height, width = images.shape
            encoded = self.encoder(images.reshape(batch * frames, colors, height, width))
            encoded = encoded.reshape(batch, frames, self.channels, *encoded.shape[-2:])
            current, previous, older = encoded.unbind(dim=1)
            if self.mode == "aligned":
                aligned_previous, aligned_older = self._aligned_history(current, previous, older)
            else:
                aligned_previous, aligned_older = previous, older
        features = self.fusion(torch.cat(
            (current, previous, older, aligned_previous, aligned_older), dim=1
        ))
        return {
            "heatmap": self.heatmap(features),
            "offset": self.offset(features),
            "log_wh": self.log_wh(features),
        }
