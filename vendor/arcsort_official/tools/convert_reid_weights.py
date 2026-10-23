"""One-time conversion of FastReID weights to TorchScript format.

Loads a FastReID SBS model using the FastReID copy bundled with Deep-OC-SORT
(clone github.com/GerardMaggiolino/Deep-OC-SORT to external/Deep-OC-SORT, or
point DEEP_OCSORT_ROOT at a checkout), traces it with TorchScript, and saves
a standalone .pt file that can be loaded without any FastReID dependency.

This script is the ONLY place that imports FastReID. The runtime code
(arcsort/reid.py) loads only the traced TorchScript model.

Usage:
    PYTHONPATH=. python tools/convert_reid_weights.py \
        --input weights/dance_sbs_S50.pth \
        --output weights/dance_sbs_S50_traced.pt \
        --config dancetrack

    # For MOT17:
    PYTHONPATH=. python tools/convert_reid_weights.py \
        --input weights/mot17_sbs_S50.pth \
        --output weights/mot17_sbs_S50_traced.pt \
        --config mot17
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from loguru import logger

from arcsort.utils import setup_logging

# Reference FastReID paths
_DEEP_OC_SORT = Path(os.environ.get(
    "DEEP_OCSORT_ROOT",
    Path(__file__).resolve().parent.parent / "external" / "Deep-OC-SORT",
))
_FASTREID_CONFIGS = _DEEP_OC_SORT / "external" / "fast_reid" / "configs"

CONFIG_MAP = {
    "dancetrack": _FASTREID_CONFIGS / "dancetrack" / "sbs_S50.yml",
    "mot17": _FASTREID_CONFIGS / "MOT17" / "sbs_S50.yml",
    "mot20": _FASTREID_CONFIGS / "MOT20" / "sbs_S50.yml",
}


def convert(
    input_path: str,
    output_path: str,
    config_name: str,
) -> None:
    """Convert FastReID weights to TorchScript.

    Args:
        input_path: Path to FastReID .pth weights.
        output_path: Path to save TorchScript .pt model.
        config_name: Config preset (dancetrack, mot17, mot20).
    """
    if config_name not in CONFIG_MAP:
        raise ValueError(f"Unknown config '{config_name}'. Options: {list(CONFIG_MAP.keys())}")

    config_file = CONFIG_MAP[config_name]
    if not config_file.exists():
        raise FileNotFoundError(
            f"FastReID config not found: {config_file}. "
            f"Clone Deep-OC-SORT to external/Deep-OC-SORT or set DEEP_OCSORT_ROOT."
        )

    # Add FastReID to path (only for this conversion script)
    ext_path = str(_DEEP_OC_SORT / "external")
    root_path = str(_DEEP_OC_SORT)
    if ext_path not in sys.path:
        sys.path.insert(0, ext_path)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)

    from fast_reid.fastreid.config import get_cfg
    from fast_reid.fastreid.modeling.meta_arch import build_model
    from fast_reid.fastreid.utils.checkpoint import Checkpointer

    logger.info(f"Loading FastReID model from {input_path}")
    logger.info(f"Using config: {config_file}")

    cfg = get_cfg()
    cfg.merge_from_file(str(config_file))
    cfg.merge_from_list(["MODEL.WEIGHTS", input_path])
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.freeze()

    model = build_model(cfg)
    model.eval()
    model.cuda()
    Checkpointer(model).load(input_path)
    model.half()

    # Trace with a dummy input matching expected crop size
    # Input: (batch, 3, H=384, W=128) in FP16
    dummy = torch.randn(1, 3, 384, 128).cuda().half()
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)

    # Verify
    with torch.no_grad():
        out_orig = model(dummy)
        out_traced = traced(dummy)
    diff = (out_orig - out_traced).abs().max().item()
    logger.info(f"Output shape: {out_traced.shape}, max trace diff: {diff:.6f}")

    traced.save(output_path)
    logger.info(f"TorchScript model saved to {output_path}")

    # Verify standalone load
    model2 = torch.jit.load(output_path)
    with torch.no_grad():
        out2 = model2(dummy)
    diff2 = (out_orig - out2).abs().max().item()
    logger.info(f"Reload verification: max diff = {diff2:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert FastReID weights to TorchScript (one-time)"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to FastReID .pth weights (e.g. weights/dance_sbs_S50.pth)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to save TorchScript model (e.g. weights/dance_sbs_S50_traced.pt)",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        choices=list(CONFIG_MAP.keys()),
        help="FastReID config preset",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    convert(args.input, args.output, args.config)


if __name__ == "__main__":
    main()
