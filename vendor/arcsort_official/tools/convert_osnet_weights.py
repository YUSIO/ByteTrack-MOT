"""Convert an OSNet ReID state dict to a standalone TorchScript module.

Wraps the OSNet backbone (architecture vendored in third_party/osnet/osnet.py,
from KaiyangZhou/deep-person-reid, MIT license) with its preprocessing so
the traced module matches EmbeddingComputer's input contract: float32 RGB
crops in [0, 255], CHW. Normalization (/255 + ImageNet mean/std) happens
inside the traced graph — unlike FastReID weights, which take raw [0, 255].

The input state dict must be a sanitized tensors-only file (see the
weights_only extraction step); this tool loads it with weights_only=True.

Usage:
    PYTHONPATH=. python tools/convert_osnet_weights.py \
        --input weights/sportsmot/sports_osnet_clean.pth \
        --output weights/sportsmot/sports_osnet_traced.pt \
        --num-classes 616

Then in the tracker config:
    reid:
      weights: "weights/sportsmot/sports_osnet_traced.pt"
      crop_size: [128, 256]   # (W, H) -- OSNet standard input
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "osnet"))


class OSNetWithPreproc(torch.nn.Module):
    """OSNet wrapped with /255 + ImageNet normalization."""

    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0
        x = (x - self.mean) / self.std
        return self.backbone(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace OSNet ReID weights")
    parser.add_argument("--input", required=True,
                        help="Sanitized tensors-only OSNet state dict")
    parser.add_argument("--output", required=True,
                        help="Output TorchScript .pt path")
    parser.add_argument("--num-classes", type=int, default=616,
                        help="Classifier size in the checkpoint (unused at "
                             "inference, needed to load the state dict)")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    args = parser.parse_args()

    from osnet import osnet_x1_0

    backbone = osnet_x1_0(
        num_classes=args.num_classes, pretrained=False, loss="softmax"
    )
    state = torch.load(args.input, map_location="cpu", weights_only=True)
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    if missing or unexpected:
        logger.warning(f"missing={missing} unexpected={unexpected}")
    backbone.eval()

    model = OSNetWithPreproc(backbone).cuda().half().eval()
    example = torch.rand(2, 3, args.height, args.width).cuda().half() * 255

    with torch.no_grad():
        traced = torch.jit.trace(model, example)
        out = traced(example)
    logger.info(f"Traced OK; output shape {tuple(out.shape)}")
    assert out.shape[1] == 512, "expected 512-d OSNet features"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    traced.save(args.output)
    logger.info(f"Saved TorchScript model to {args.output}")


if __name__ == "__main__":
    main()
