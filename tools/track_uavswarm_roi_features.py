#!/usr/bin/env python3
"""Track scored UAVSwarm detections with a fixed frozen-ROI primary cost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from uavswarm_yolo_roi import (
    assert_record_matches_detections,
    load_detections,
    load_sequence_record,
    read_sequence_info,
    sha256,
)
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.byte_tracker import BYTETracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--detections-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-cache-root", type=Path)
    parser.add_argument("--appearance-weight", type=float, required=True)
    parser.add_argument(
        "--appearance-permutation-seed",
        type=int,
        help="Deterministically permute track-history features per primary-association frame.",
    )
    parser.add_argument(
        "--appearance-constant-cost",
        type=float,
        help="Use a constant visual cost as a negative control instead of feature cosine.",
    )
    parser.add_argument("--track-thresh", type=float, default=0.6)
    parser.add_argument("--det-thresh", type=float, default=None)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--match-thresh", type=float, default=0.9)
    parser.add_argument("--min-box-area", type=float, default=100.0)
    parser.add_argument("--aspect-ratio-thresh", type=float, default=3.0)
    parser.add_argument("--mot20", action="store_true")
    parser.add_argument("--sequences", nargs="+", metavar="SEQUENCE")
    return parser.parse_args()


def write_sequence_results(path: Path, rows: list[tuple[int, int, float, float, float, float, float]]) -> None:
    with path.open("w", encoding="utf-8") as destination:
        for frame, track_id, x, y, width, height, score in rows:
            destination.write(
                f"{frame},{track_id},{round(x, 1)},{round(y, 1)},"
                f"{round(width, 1)},{round(height, 1)},{round(score, 2)},-1,-1,-1\n"
            )


def track_sequence(sequence_dir: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    info = read_sequence_info(sequence_dir)
    det_path = args.detections_root / str(info["name"]) / "det.txt"
    detections = load_detections(det_path)
    features_by_frame: dict[int, np.ndarray] = {}
    feature_dimension = None
    if args.appearance_weight > 0.0:
        record_path = args.feature_cache_root / f"{info['name']}.npz"
        record = load_sequence_record(record_path)
        features_by_frame = assert_record_matches_detections(record, info, detections, record_path)
        feature_dimension = int(record["features"].shape[1])

    BaseTrack._count = 0
    tracker = BYTETracker(args, frame_rate=int(info["frame_rate"]))
    result_rows = []
    input_count = 0
    feature_frames = 0
    for frame in range(1, int(info["frames"]) + 1):
        frame_detections = detections.get(frame)
        if frame_detections is None:
            outputs = np.empty((0, 5), dtype=np.float32)
        else:
            outputs = frame_detections.copy()
        input_count += len(outputs)
        appearance_features = None
        if len(outputs) and args.appearance_weight > 0.0:
            appearance_features = features_by_frame.get(frame)
            if appearance_features is None or len(appearance_features) != len(outputs):
                raise ValueError(f"{info['name']} frame {frame}: feature-cache alignment failure")
            feature_frames += 1
        online_targets = tracker.update(
            outputs,
            (int(info["height"]), int(info["width"])),
            (int(info["height"]), int(info["width"])),
            appearance_features=appearance_features,
        )
        for target in online_targets:
            x, y, width, height = target.tlwh
            is_vertical = width / height > args.aspect_ratio_thresh
            if width * height > args.min_box_area and not is_vertical:
                result_rows.append((frame, target.track_id, x, y, width, height, target.score))
    output_path = output_dir / f"{info['name']}.txt"
    write_sequence_results(output_path, result_rows)
    return {
        "sequence": info["name"],
        "frames": int(info["frames"]),
        "input_detection_rows": input_count,
        "output_tracking_rows": len(result_rows),
        "det_path": str(det_path.resolve()),
        "det_sha256": sha256(det_path),
        "feature_frames": feature_frames,
        "feature_dimension": feature_dimension,
        "output_sha256": sha256(output_path),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.appearance_weight <= 1.0:
        raise ValueError("appearance-weight must be in [0, 1]")
    if args.appearance_weight > 0.0 and args.feature_cache_root is None:
        raise ValueError("feature-cache-root is required when appearance-weight is positive")
    if args.appearance_permutation_seed is not None and args.appearance_weight <= 0.0:
        raise ValueError("appearance-permutation-seed requires a positive appearance-weight")
    if args.appearance_constant_cost is not None:
        if args.appearance_weight <= 0.0:
            raise ValueError("appearance-constant-cost requires a positive appearance-weight")
        if not 0.0 <= args.appearance_constant_cost <= 1.0:
            raise ValueError("appearance-constant-cost must be in [0, 1]")
        if args.appearance_permutation_seed is not None:
            raise ValueError("appearance-constant-cost cannot be combined with feature permutation")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    split_root = args.dataset_root / args.split
    available = [path for path in sorted(split_root.glob("UAVSwarm-*")) if path.is_dir()]
    names = {path.name for path in available}
    selected_names = list(dict.fromkeys(args.sequences)) if args.sequences else [path.name for path in available]
    unknown = sorted(set(selected_names) - names)
    if unknown:
        raise ValueError(f"unknown {args.split} sequences: {', '.join(unknown)}")
    if args.feature_cache_root is not None and not args.feature_cache_root.is_dir():
        raise FileNotFoundError(f"missing feature-cache root: {args.feature_cache_root}")

    args.output_dir.mkdir(parents=True)
    summaries = [track_sequence(split_root / name, args.output_dir, args) for name in selected_names]
    summary = {
        "parameters": {
            "track_thresh": args.track_thresh,
            "det_thresh": args.det_thresh if args.det_thresh is not None else args.track_thresh + 0.1,
            "track_buffer": args.track_buffer,
            "match_thresh": args.match_thresh,
            "min_box_area": args.min_box_area,
            "aspect_ratio_thresh": args.aspect_ratio_thresh,
            "mot20": args.mot20,
            "appearance_weight": args.appearance_weight,
            "appearance_policy": "single fixed convex cost for every primary-association pair; no gate",
            "appearance_permutation_seed": args.appearance_permutation_seed,
            "appearance_permutation_policy": (
                "per-frame deterministic permutation of track-history features"
                if args.appearance_permutation_seed is not None
                else None
            ),
            "appearance_constant_cost": args.appearance_constant_cost,
            "appearance_null_policy": (
                "constant visual cost negative control"
                if args.appearance_constant_cost is not None
                else None
            ),
        },
        "detector_cache_root": str(args.detections_root.resolve()),
        "feature_cache_root": str(args.feature_cache_root.resolve()) if args.feature_cache_root else None,
        "sequences": summaries,
        "totals": {
            "sequences": len(summaries),
            "frames": sum(int(row["frames"]) for row in summaries),
            "input_detection_rows": sum(int(row["input_detection_rows"]) for row in summaries),
            "output_tracking_rows": sum(int(row["output_tracking_rows"]) for row in summaries),
            "feature_frames": sum(int(row["feature_frames"]) for row in summaries),
        },
    }
    with (args.output_dir / "tracking_summary.json").open("w", encoding="utf-8") as destination:
        json.dump(summary, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")


if __name__ == "__main__":
    main()
