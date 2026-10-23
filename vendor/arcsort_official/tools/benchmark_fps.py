"""Benchmark tracker FPS on cached detections.

Measures wall-clock time for tracker.update() calls across all frames
of a sequence or dataset split. Reports both tracker-only FPS (no ReID)
and full-pipeline FPS (with ReID embedding computation).

Usage:
    # Tracker-only (no frame loading, no ReID):
    python tools/benchmark_fps.py \
        --config arcsort/configs/dancetrack.yaml \
        --detections detections/dancetrack_val_yolox_x_norm/ \
        --no-reid

    # Full pipeline (with ReID + CMC):
    python tools/benchmark_fps.py \
        --config arcsort/configs/dancetrack.yaml \
        --detections detections/dancetrack_val_yolox_x_norm/ \
        --frames data/DanceTrack/val \
        --cmc-cache cmc_cache/dancetrack_val/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from arcsort.tracker import ARCTracker, load_config
from arcsort.utils import setup_logging


def load_detections(det_file: Path, min_confidence: float = 0.0) -> dict[int, np.ndarray]:
    """Load cached detections from MOT-format file."""
    if not det_file.exists():
        raise FileNotFoundError(f"Detection file not found: {det_file}")

    data = np.loadtxt(det_file, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] == 0:
        return {}

    if min_confidence > 0:
        data = data[data[:, 6] >= min_confidence]
        if data.shape[0] == 0:
            return {}

    detections_by_frame: dict[int, np.ndarray] = {}
    frame_ids = data[:, 0].astype(int)
    for frame_id in np.unique(frame_ids):
        mask = frame_ids == frame_id
        fd = data[mask]
        x1, y1, w, h, conf = fd[:, 2], fd[:, 3], fd[:, 4], fd[:, 5], fd[:, 6]
        detections_by_frame[frame_id] = np.column_stack([
            x1, y1, x1 + w, y1 + h, conf, np.zeros(len(conf))
        ])
    return detections_by_frame


def benchmark_sequence(
    tracker: ARCTracker,
    detections_by_frame: dict[int, np.ndarray],
    frames_dir: Path | None = None,
    cmc_cache: dict[int, np.ndarray] | None = None,
    use_reid: bool = False,
) -> tuple[int, float, float]:
    """Benchmark a single sequence.

    Returns:
        Tuple of (total_frames, tracker_time_sec, frame_load_time_sec).
    """
    tracker.reset()
    if not detections_by_frame:
        return 0, 0.0, 0.0

    max_frame = max(detections_by_frame.keys())
    tracker_time = 0.0
    frame_load_time = 0.0
    total_frames = 0

    for frame_id in range(1, max_frame + 1):
        dets = detections_by_frame.get(frame_id, np.empty((0, 6)))

        frame = None
        cmc_affine = None

        if cmc_cache is not None:
            cmc_affine = cmc_cache.get(frame_id)

        if frames_dir is not None and use_reid:
            t0 = time.perf_counter()
            # DanceTrack pads frame names to 8 digits, SportsMOT to 6
            frame_path = frames_dir / f"{frame_id:08d}.jpg"
            if not frame_path.exists():
                frame_path = frames_dir / f"{frame_id:06d}.jpg"
            if frame_path.exists():
                frame = cv2.imread(str(frame_path))
            else:
                logger.warning(f"Frame not found: {frame_path} — ReID skipped")
            frame_load_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        tracker.update(dets, frame=frame, cmc_affine=cmc_affine)
        tracker_time += time.perf_counter() - t0
        total_frames += 1

    return total_frames, tracker_time, frame_load_time


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ARC-SORT FPS")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--detections", type=str, required=True)
    parser.add_argument("--frames", type=str, default=None)
    parser.add_argument("--cmc-cache", type=str, default=None)
    parser.add_argument(
        "--no-reid", action="store_true",
        help="Disable ReID even if config enables it (tracker-only benchmark)",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup sequences to skip")
    args = parser.parse_args()

    setup_logging(False)

    config = load_config(args.config)
    if args.no_reid:
        config["use_reid"] = False
        config["use_gated_reid"] = False

    use_reid = config.get("use_reid", False) or config.get("use_gated_reid", False)

    det_dir = Path(args.detections)
    frames_base = Path(args.frames) if args.frames else None
    cmc_cache_dir = Path(args.cmc_cache) if args.cmc_cache else None

    det_files = sorted(det_dir.glob("*.txt"))
    if not det_files:
        raise FileNotFoundError(f"No .txt files in {det_dir}")

    tracker = ARCTracker(config)

    total_frames = 0
    total_tracker_time = 0.0
    total_frame_load_time = 0.0
    seq_results = []

    for i, det_file in enumerate(det_files):
        seq_name = det_file.stem

        seq_cmc_cache = None
        seq_frames_dir = None

        if cmc_cache_dir is not None:
            cmc_file = cmc_cache_dir / f"{seq_name}.npy"
            if cmc_file.exists():
                seq_cmc_cache = np.load(str(cmc_file), allow_pickle=True).item()

        if frames_base is not None:
            seq_frames_dir = frames_base / seq_name / "img1"
            if not seq_frames_dir.exists():
                seq_frames_dir = None

        min_conf = config.get("track_low_thresh", 0.1)
        dets = load_detections(det_file, min_confidence=min_conf)

        n_frames, t_tracker, t_load = benchmark_sequence(
            tracker, dets, frames_dir=seq_frames_dir,
            cmc_cache=seq_cmc_cache, use_reid=use_reid,
        )

        is_warmup = i < args.warmup
        tag = " (warmup)" if is_warmup else ""

        if n_frames > 0:
            seq_fps = n_frames / t_tracker
            full_fps = n_frames / (t_tracker + t_load) if t_load > 0 else seq_fps
            logger.info(
                f"  {seq_name}: {n_frames} frames, "
                f"tracker {seq_fps:.1f} FPS, "
                f"full {full_fps:.1f} FPS{tag}"
            )
            seq_results.append({
                "name": seq_name, "frames": n_frames,
                "tracker_time": t_tracker, "load_time": t_load,
                "warmup": is_warmup,
            })

            if not is_warmup:
                total_frames += n_frames
                total_tracker_time += t_tracker
                total_frame_load_time += t_load

    # Summary
    print("\n" + "=" * 60)
    print(f"Config: {args.config}")
    print(f"ReID: {'enabled' if use_reid else 'disabled'}")
    print(f"Sequences: {len(det_files)} ({args.warmup} warmup)")
    print(f"Total frames (excl. warmup): {total_frames}")
    print(f"Tracker time: {total_tracker_time:.2f}s")
    if total_frame_load_time > 0:
        print(f"Frame load time: {total_frame_load_time:.2f}s")
    print("-" * 60)
    tracker_fps = total_frames / total_tracker_time if total_tracker_time > 0 else 0
    print(f"Tracker-only FPS: {tracker_fps:.1f}")
    if total_frame_load_time > 0:
        full_fps = total_frames / (total_tracker_time + total_frame_load_time)
        print(f"Full pipeline FPS: {full_fps:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
