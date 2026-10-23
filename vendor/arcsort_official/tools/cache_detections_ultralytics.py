"""Cache detections from Ultralytics models on DanceTrack sequences.

Runs an Ultralytics YOLO/RT-DETR model on each frame and saves detections
in MOT format (one .txt per sequence) for use with run_cached.py.

Supports: YOLOv8, YOLOv9, YOLOv10, YOLOv11, RT-DETR, and any model
loadable via ultralytics.YOLO().

Detection output format (per line):
    frame_id, -1, x1, y1, w, h, confidence, -1, -1, -1

Usage:
    PYTHONPATH=. python tools/cache_detections_ultralytics.py \
        --model yolo11x.pt \
        --frames data/DanceTrack/val \
        --output detections/dancetrack_val_yolo11x/ \
        --img-size 1440 \
        --conf-thresh 0.01 \
        --classes 0

    # Multiple models at once:
    for m in yolov8x yolov9e yolo11x rtdetr-x; do
        PYTHONPATH=. python tools/cache_detections_ultralytics.py \
            --model ${m}.pt --frames data/DanceTrack/val \
            --output detections/dancetrack_val_${m}/ \
            --img-size 1440 --conf-thresh 0.01 --classes 0
    done
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from loguru import logger

from arcsort.utils import setup_logging


def cache_sequence(
    model,
    frames_dir: Path,
    output_file: Path,
    img_size: int,
    conf_thresh: float,
    classes: list[int] | None,
) -> int:
    """Run detection on a single sequence and save to MOT format.

    Args:
        model: Loaded Ultralytics model.
        frames_dir: Path to sequence frames (e.g., .../dancetrack0004/img1/).
        output_file: Path to save detection .txt file.
        img_size: Inference image size (longest side).
        conf_thresh: Minimum confidence to save.
        classes: List of class IDs to keep (0=person). None for all.

    Returns:
        Number of detections saved.
    """
    frame_files = sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("*.png"))
    if not frame_files:
        logger.warning(f"No frames found in {frames_dir}")
        return 0

    lines = []
    total_dets = 0

    for frame_path in frame_files:
        # Extract frame ID from filename (e.g., 00000001.jpg -> 1)
        frame_id = int(frame_path.stem)

        # Run inference
        results = model.predict(
            str(frame_path),
            imgsz=img_size,
            conf=conf_thresh,
            classes=classes,
            verbose=False,
            half=True,
        )

        if len(results) == 0 or results[0].boxes is None:
            continue

        boxes = results[0].boxes
        if boxes.shape[0] == 0:
            continue

        # Get xyxy + confidence
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[i]
            w = x2 - x1
            h = y2 - y1
            conf = confs[i]
            line = f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.6f},-1,-1,-1"
            lines.append(line)
            total_dets += 1

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")

    return total_dets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache detections from Ultralytics models"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name or path (e.g., yolo11x.pt, yolov8x.pt, rtdetr-x.pt)",
    )
    parser.add_argument(
        "--frames", type=str, required=True,
        help="Root directory of dataset frames. Expected structure: "
             "<frames>/<seq_name>/img1/<frame_id>.jpg",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Directory to save cached detection .txt files",
    )
    parser.add_argument(
        "--img-size", type=int, default=1440,
        help="Inference image size (longest side). Default 1440.",
    )
    parser.add_argument(
        "--conf-thresh", type=float, default=0.01,
        help="Minimum confidence to save. Default 0.01 (save almost all).",
    )
    parser.add_argument(
        "--classes", type=int, nargs="+", default=[0],
        help="Class IDs to keep. Default: 0 (person). Use -1 for all.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    from ultralytics import YOLO

    # Resolve model path: check weights/ directory first, then download
    model_path = args.model
    weights_dir = Path(__file__).parent.parent / "weights"
    weights_candidate = weights_dir / Path(args.model).name
    if weights_candidate.exists():
        model_path = str(weights_candidate)
    elif not Path(model_path).exists():
        # Will be downloaded by Ultralytics to current dir -- move to weights/
        logger.info(f"Downloading model: {args.model}")
        weights_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Move downloaded weights to weights/ if they landed in cwd
    cwd_weights = Path(Path(args.model).name)
    if cwd_weights.exists() and cwd_weights.parent == Path("."):
        dest = weights_dir / cwd_weights.name
        if not dest.exists():
            cwd_weights.rename(dest)
            logger.info(f"Moved weights to {dest}")

    logger.info(f"Model ready: {args.model}")

    frames_base = Path(args.frames)
    if not frames_base.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_base}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = args.classes if args.classes != [-1] else None

    # Find all sequences
    seq_dirs = sorted([
        d for d in frames_base.iterdir()
        if d.is_dir() and (d / "img1").exists()
    ])
    if not seq_dirs:
        raise FileNotFoundError(f"No sequences found in {frames_base}")

    logger.info(f"Found {len(seq_dirs)} sequences")

    total_dets = 0
    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        frames_dir = seq_dir / "img1"
        output_file = out_dir / f"{seq_name}.txt"

        n = cache_sequence(
            model, frames_dir, output_file,
            img_size=args.img_size,
            conf_thresh=args.conf_thresh,
            classes=classes,
        )
        total_dets += n
        num_frames = len(list(frames_dir.glob("*.jpg")))
        logger.info(f"  {seq_name}: {num_frames} frames, {n} detections")

    logger.info(f"Total: {total_dets} detections saved to {out_dir}")


if __name__ == "__main__":
    main()
