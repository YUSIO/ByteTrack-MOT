"""Pre-compute and cache CMC affine transforms for all sequences.

Runs the CameraMotionCompensator on raw frame images and saves the
per-frame 2x3 affine matrices as .npy files. These cached transforms
can then be loaded by run_cached.py via --cmc-cache, eliminating the
need to load frame images during tracking (~8 min -> ~0 overhead).

Usage:
    python tools/cache_cmc.py \
        --frames data/DanceTrack/val \
        --output cmc_cache/dancetrack_val/ \
        --config arcsort/configs/sportsmot.yaml

Output format:
    One .npy file per sequence containing a dict:
        {frame_id: np.ndarray of shape (2, 3)}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from arcsort.cmc import CameraMotionCompensator
from arcsort.tracker import load_config
from arcsort.utils import setup_logging


def cache_sequence_cmc(
    frames_dir: Path,
    cmc: CameraMotionCompensator,
) -> dict[int, np.ndarray]:
    """Compute CMC affine transforms for all frames in a sequence.

    Args:
        frames_dir: Path to frame images (e.g., .../dancetrack0004/img1/).
        cmc: CameraMotionCompensator instance (will be reset).

    Returns:
        Dict mapping frame_id to 2x3 affine matrix.
    """
    cmc.reset()

    frame_files = sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        return {}

    affines: dict[int, np.ndarray] = {}

    for frame_path in frame_files:
        frame_id = int(frame_path.stem)
        frame = cv2.imread(str(frame_path))
        if frame is None:
            affines[frame_id] = np.eye(2, 3, dtype=np.float64)
            continue

        affine = cmc.apply(frame)
        affines[frame_id] = affine

    return affines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute and cache CMC affine transforms"
    )
    parser.add_argument(
        "--frames", type=str, required=True,
        help="Base directory containing sequence frame images. "
             "Expected structure: <frames>/<seq_name>/img1/*.jpg",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Directory to write cached CMC .npy files",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Tracker config YAML (reads CMC parameters from it). "
             "If not provided, uses default CMC config.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    frames_base = Path(args.frames)
    if not frames_base.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_base}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmc_config = {}
    if args.config:
        config = load_config(args.config)
        cmc_config = config.get("cmc", {})

    cmc = CameraMotionCompensator(cmc_config)

    seq_dirs = sorted([
        d for d in frames_base.iterdir()
        if d.is_dir() and (d / "img1").exists()
    ])

    if not seq_dirs:
        raise FileNotFoundError(
            f"No sequences found in {frames_base} "
            f"(expected <seq_name>/img1/ structure)"
        )

    logger.info(f"Caching CMC for {len(seq_dirs)} sequences")

    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        img_dir = seq_dir / "img1"
        logger.info(f"Processing {seq_name}...")

        affines = cache_sequence_cmc(img_dir, cmc)

        out_file = out_dir / f"{seq_name}.npy"
        np.save(str(out_file), affines, allow_pickle=True)
        logger.info(f"  {len(affines)} frames -> {out_file}")

    logger.info(f"CMC cache written to {out_dir}")


if __name__ == "__main__":
    main()
