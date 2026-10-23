"""Pre-compute ReID embeddings for cached detections.

For every sequence in a detection cache directory, loads the frame
images, computes L2-normalized ReID embeddings for every detection
above a confidence floor, and saves one compressed .npz per sequence.
Tracker runs can then use --emb-cache (tools/run_cached.py) to iterate
on association parameters with no GPU and no frame I/O.

This is the GPU-heavy half of the decoupled pipeline and is designed to
run once per (dataset split, detector) pair — locally or in a cloud
notebook (Kaggle/Colab) — with the resulting cache downloaded for local
CPU-only tracking runs.

Cache format per sequence (.npz):
    boxes_{frame_id}: (N, 4) float32 [x1, y1, x2, y2]
    embs_{frame_id}:  (N, 2048) float16, L2-normalized

Usage:
    PYTHONPATH=. python tools/cache_embeddings.py \
        --detections detections/dancetrack_val_yolox_x_norm/ \
        --frames data/DanceTrack/val \
        --config arcsort/configs/dancetrack.yaml \
        --output emb_cache/dancetrack_val_yolox_x_norm/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcsort.reid import EmbeddingComputer
from arcsort.tracker import DEFAULT_CONFIG, load_config
from tools.run_cached import load_detections


def resolve_frames_dir(frames_base: Path, seq_name: str) -> Path | None:
    """Resolve a sequence's frame directory, with MOT17 variant fallback.

    Args:
        frames_base: Dataset split root (e.g. data/DanceTrack/val).
        seq_name: Sequence name from the detection file stem.

    Returns:
        Path to the img1 directory, or None if not found.
    """
    seq_frames = frames_base / seq_name / "img1"
    if seq_frames.exists():
        return seq_frames
    for suffix in ["SDP", "FRCNN", "DPM"]:
        alt = frames_base / f"{seq_name}-{suffix}" / "img1"
        if alt.exists():
            return alt
    return None


def load_frame(frames_dir: Path, frame_id: int) -> np.ndarray | None:
    """Load one frame image, trying common naming conventions."""
    for fmt in (f"{frame_id:08d}.jpg", f"{frame_id:06d}.jpg"):
        frame_path = frames_dir / fmt
        if frame_path.exists():
            return cv2.imread(str(frame_path))
    return None


def cache_sequence(
    computer: EmbeddingComputer,
    detections_by_frame: dict[int, np.ndarray],
    frames_dir: Path,
    out_file: Path,
) -> int:
    """Compute and save embeddings for every frame of one sequence.

    Args:
        computer: EmbeddingComputer with the ReID model.
        detections_by_frame: frame_id -> (N, 6) detections.
        frames_dir: Directory with the sequence's frame images.
        out_file: Output .npz path.

    Returns:
        Number of embeddings computed.
    """
    arrays: dict[str, np.ndarray] = {}
    total = 0
    for frame_id in sorted(detections_by_frame):
        dets = detections_by_frame[frame_id]
        if dets.shape[0] == 0:
            continue
        frame = load_frame(frames_dir, frame_id)
        if frame is None:
            logger.warning(f"Frame {frame_id} not found in {frames_dir}; skipping")
            continue
        boxes = dets[:, :4].astype(np.float32)
        embs = computer.compute_embeddings(frame, boxes)
        arrays[f"boxes_{frame_id}"] = boxes
        arrays[f"embs_{frame_id}"] = embs.astype(np.float16)
        total += boxes.shape[0]

    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_file, **arrays)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute ReID embeddings for cached detections"
    )
    parser.add_argument(
        "--detections", type=str, required=True,
        help="Directory of cached detection .txt files (one per sequence)",
    )
    parser.add_argument(
        "--frames", type=str, required=True,
        help="Dataset split root containing <seq>/img1/ frame images",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Tracker config YAML (for reid weights/crop_size). "
             "Defaults to the built-in reid defaults.",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for per-sequence .npz embedding caches",
    )
    parser.add_argument(
        "--min-conf", type=float, default=0.1,
        help="Skip detections below this confidence (default 0.1; must be "
             "<= the lowest track_high_thresh you intend to sweep)",
    )
    args = parser.parse_args()
    logger.enable("arcsort")

    reid_cfg = dict(DEFAULT_CONFIG["reid"])
    if args.config:
        config = load_config(args.config)
        reid_cfg.update(config.get("reid") or {})

    computer = EmbeddingComputer(
        weights_path=reid_cfg["weights"],
        crop_size=tuple(reid_cfg["crop_size"]),
    )

    det_dir = Path(args.detections)
    frames_base = Path(args.frames)
    out_dir = Path(args.output)
    det_files = sorted(det_dir.glob("*.txt"))
    if not det_files:
        raise FileNotFoundError(f"No .txt detection files found in {det_dir}")

    logger.info(f"Caching embeddings for {len(det_files)} sequences -> {out_dir}")
    for det_file in det_files:
        seq_name = det_file.stem
        out_file = out_dir / f"{seq_name}.npz"
        if out_file.exists():
            logger.info(f"{seq_name}: cache exists, skipping")
            continue

        frames_dir = resolve_frames_dir(frames_base, seq_name)
        if frames_dir is None:
            logger.warning(f"{seq_name}: frames not found under {frames_base}, skipping")
            continue

        start = time.time()
        detections_by_frame = load_detections(det_file, min_confidence=args.min_conf)
        total = cache_sequence(computer, detections_by_frame, frames_dir, out_file)
        logger.info(
            f"{seq_name}: {total} embeddings in {time.time() - start:.1f}s"
        )

    logger.info(f"Done. Embedding caches written to {out_dir}")


if __name__ == "__main__":
    main()
