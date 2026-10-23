"""Extract a tensors-only state dict from a training checkpoint.

Training checkpoints (e.g. Deep-EIoU's torchreid ``sports_model.pth.tar-60``)
pickle optimizer state and numpy scalars, so they only load with
``weights_only=False``, which can execute arbitrary code. Run this once on a
file downloaded from the official release; downstream tools then load the
output with ``weights_only=True``.

Usage:
    python tools/extract_state_dict.py \
        --input sports_model.pth.tar-60 \
        --output weights/sportsmot/sports_osnet_clean.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a tensors-only state dict")
    parser.add_argument("--input", required=True, help="Checkpoint from a trusted source")
    parser.add_argument("--output", required=True, help="Output .pth (tensors only)")
    parser.add_argument("--key", default="state_dict",
                        help="Checkpoint key holding the state dict (default: state_dict)")
    args = parser.parse_args()

    logger.warning(f"Loading {args.input} with weights_only=False: only use official releases")
    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    state = ckpt[args.key] if isinstance(ckpt, dict) and args.key in ckpt else ckpt
    state = {k.removeprefix("module."): v for k, v in state.items() if torch.is_tensor(v)}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    logger.info(f"Saved {len(state)} tensors to {args.output}")


if __name__ == "__main__":
    main()
