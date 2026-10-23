"""Render side-by-side comparison videos for the multi-tracker benchmark.

Produces, for one showcase sequence:
  - one full-resolution annotated mp4 per tracker (boxes, ids, traces, and a
    banner with tracker name / HOTA / FPS), and
  - one tiled 2x4 grid video (7 trackers + a raw-detections tile) so all
    trackers can be compared frame-locked on the same footage.

The showcase sequence defaults to the one where ARC-SORT's per-sequence HOTA
lead over the best baseline is largest (from per_seq_hota.csv).

Usage:
    PYTHONPATH=. python tools/make_comparison_video.py \
        --results results/comparison_sportsmot_val \
        --data data/SportsMOT/val \
        --out demo_vids
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcsort.utils import setup_logging
from arcsort.visualizer import Visualizer
from tools.run_cached import load_detections

VIDEO_FPS = 25.0
BANNER_H = 44

# Grid layout: strongest trackers on top, raw detections tile last.
GRID_ORDER = [
    "arcsort", "deepocsort", "strongsort", "botsort",
    "ocsort", "bytetrack", "sort", "__detections__",
]
TILE_W, TILE_H = 640, 360


def load_comparison(results_dir: Path) -> dict[str, dict]:
    rows = {}
    with open(results_dir / "comparison.csv") as f:
        for row in csv.DictReader(f):
            rows[row["tracker"]] = row
    return rows


def pick_showcase(results_dir: Path) -> str:
    """Sequence with the largest ARC-SORT HOTA lead over the best baseline."""
    best_seq, best_gap = None, -1e9
    with open(results_dir / "per_seq_hota.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ours = float(row["arcsort"])
            rivals = [
                float(v) for k, v in row.items()
                if k not in ("sequence", "arcsort") and v
            ]
            gap = ours - max(rivals)
            if gap > best_gap:
                best_gap, best_seq = gap, row["sequence"]
    logger.info(f"Showcase sequence: {best_seq} (ARC-SORT lead {best_gap:+.2f} HOTA)")
    return best_seq


def load_tracks(result_file: Path) -> dict[int, np.ndarray]:
    """MOT result file -> frame_id -> (M,5) [x1,y1,x2,y2,tid]."""
    by_frame: dict[int, list] = defaultdict(list)
    data = np.loadtxt(result_file, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    for row in data:
        fid = int(row[0])
        x, y, w, h = row[2:6]
        by_frame[fid].append([x, y, x + w, y + h, int(row[1])])
    return {fid: np.array(rows) for fid, rows in by_frame.items()}


def banner(frame: np.ndarray, text: str) -> np.ndarray:
    """Draw a dark banner strip with white text across the top."""
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], BANNER_H), (25, 25, 25), -1)
    cv2.putText(
        out, text, (12, BANNER_H - 14), cv2.FONT_HERSHEY_SIMPLEX,
        0.85, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return out


def draw_detections(frame: np.ndarray, dets: np.ndarray) -> np.ndarray:
    """Plain yellow boxes for the raw-detections tile (conf >= 0.1)."""
    out = frame.copy()
    for x1, y1, x2, y2, conf, _ in dets:
        if conf < 0.1:
            continue
        cv2.rectangle(
            out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 255), 2
        )
    return out


def make_writer(path: Path, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def main() -> None:
    parser = argparse.ArgumentParser(description="Render comparison videos")
    parser.add_argument("--results", default="results/comparison_sportsmot_val")
    parser.add_argument("--data", default="data/SportsMOT/val")
    parser.add_argument("--detections", default="detections/sportsmot_val_yolox_x")
    parser.add_argument("--out", default="demo_vids")
    parser.add_argument("--seq", default=None, help="Override showcase sequence")
    parser.add_argument("--trace-length", type=int, default=40)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    results_dir = Path(args.results)
    comparison = load_comparison(results_dir)
    seq = args.seq or pick_showcase(results_dir)

    seq_img_dir = Path(args.data) / seq / "img1"
    frame_files = sorted(seq_img_dir.glob("*.jpg"))
    if not frame_files:
        raise SystemExit(f"No frames found in {seq_img_dir}")
    first = cv2.imread(str(frame_files[0]))
    full_size = (first.shape[1], first.shape[0])

    trackers = [t for t in GRID_ORDER if t != "__detections__"]
    tracks = {t: load_tracks(results_dir / t / f"{seq}.txt") for t in trackers}
    dets_by_frame = load_detections(
        Path(args.detections) / f"{seq}.txt", min_confidence=0.1
    )

    out_dir = Path(args.out) / f"comparison_{seq}"
    # One stateful Visualizer per tracker: TraceAnnotator carries history.
    viz = {t: Visualizer(trace_length=args.trace_length) for t in trackers}
    writers = {
        t: make_writer(out_dir / f"{t}.mp4", full_size) for t in trackers
    }
    grid_writer = make_writer(
        out_dir / "grid.mp4", (TILE_W * 4, TILE_H * 2)
    )

    banners = {}
    for t in trackers:
        row = comparison[t]
        name = row["display"].replace(" (ours)", "*")
        banners[t] = f"{name}   HOTA {float(row['HOTA']):.2f}   {float(row['FPS']):.0f} FPS"

    logger.info(f"Rendering {len(frame_files)} frames of {seq} ...")
    for idx, frame_file in enumerate(frame_files, start=1):
        frame = cv2.imread(str(frame_file))
        tiles = []
        for t in GRID_ORDER:
            if t == "__detections__":
                annotated = banner(
                    draw_detections(
                        frame, dets_by_frame.get(idx, np.empty((0, 6)))
                    ),
                    "YOLOX-X detections (shared input)",
                )
            else:
                annotated = banner(
                    viz[t].annotate(
                        frame, tracks[t].get(idx, np.empty((0, 5)))
                    ),
                    banners[t],
                )
                writers[t].write(annotated)
            tiles.append(cv2.resize(annotated, (TILE_W, TILE_H)))
        top = np.hstack(tiles[:4])
        bottom = np.hstack(tiles[4:])
        grid_writer.write(np.vstack([top, bottom]))
        if idx % 250 == 0:
            logger.info(f"  {idx}/{len(frame_files)} frames")

    for w in writers.values():
        w.release()
    grid_writer.release()
    logger.info(f"Videos written to {out_dir}")


if __name__ == "__main__":
    main()
