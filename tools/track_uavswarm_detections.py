#!/usr/bin/env python3
"""Run unmodified ByteTrack association on UAVSwarm MOT-format detection files.

The source detection files are already in original-image coordinates.  Passing
the original image shape as both ``img_info`` and ``img_size`` preserves a
unit coordinate scale in ``BYTETracker.update``.  Aside from the NumPy-2
compatibility aliases replaced in the fork, association and lifecycle logic are
from the pinned upstream ByteTrack revision.
"""

import argparse
import configparser
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.byte_tracker import BYTETracker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--detections-root",
        type=Path,
        default=None,
        help=(
            "Directory containing one <sequence>/det.txt file per sequence. "
            "When omitted, uses <dataset-root>/<split>/<sequence>/det/det.txt."
        ),
    )
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--sequences", nargs="+", metavar="SEQUENCE")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-thresh", type=float, default=0.6)
    parser.add_argument(
        "--det-thresh",
        type=float,
        default=None,
        help=(
            "Score required to initialize a new track. When omitted, preserves "
            "the upstream-compatible value track_thresh + 0.1."
        ),
    )
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--match-thresh", type=float, default=0.9)
    parser.add_argument("--min-box-area", type=float, default=100.0)
    parser.add_argument("--aspect-ratio-thresh", type=float, default=1.6)
    parser.add_argument("--mot20", action="store_true")
    parser.add_argument("--affinity-checkpoint", type=Path)
    parser.add_argument("--affinity-device", default="cuda:0" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--affinity-weight", type=float, default=0.0)
    parser.add_argument("--affinity-min-probability", type=float, default=0.5)
    parser.add_argument("--affinity-apply-to", choices=("all", "tracked", "lost"), default="all",
                        help="restrict the affinity cost adjustment to Tracked or Lost track rows (attribution)")
    return parser.parse_args()


def read_sequence_info(sequence_dir):
    config = configparser.ConfigParser()
    config.read(sequence_dir / "seqinfo.ini")
    section = config["Sequence"]
    return {
        "name": section["name"],
        "frames": int(section["seqLength"]),
        "height": int(section["imHeight"]),
        "width": int(section["imWidth"]),
        "frame_rate": int(section["frameRate"]),
    }


def load_detections(path):
    detections = defaultdict(list)
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            fields = line.strip().split(",")
            if not line.strip():
                continue
            if len(fields) < 7:
                raise ValueError(f"{path}:{line_number}: expected at least 7 MOT fields")
            frame = int(float(fields[0]))
            x, y, width, height, score = (float(value) for value in fields[2:7])
            detections[frame].append((x, y, x + width, y + height, score))
    return detections


def write_sequence_results(path, rows):
    with path.open("w") as destination:
        for frame, track_id, x, y, width, height, score in rows:
            destination.write(
                f"{frame},{track_id},{round(x, 1)},{round(y, 1)},"
                f"{round(width, 1)},{round(height, 1)},{round(score, 2)},-1,-1,-1\n"
            )


def track_sequence(sequence_dir, output_dir, args):
    info = read_sequence_info(sequence_dir)
    if args.detections_root is None:
        detection_path = sequence_dir / "det" / "det.txt"
    else:
        detection_path = args.detections_root / info["name"] / "det.txt"
    if not detection_path.is_file():
        raise FileNotFoundError(f"missing detector cache for {info['name']}: {detection_path}")
    detections = load_detections(detection_path)
    BaseTrack._count = 0
    if args.affinity_checkpoint is not None:
        from track_conditioned_affinity import TrackAffinityPredictor
        args.association_adjuster = TrackAffinityPredictor(
            args.affinity_checkpoint, args.affinity_device, info["width"], info["height"],
            args.affinity_weight, args.affinity_min_probability, args.affinity_apply_to,
        )
    else:
        args.association_adjuster = None
    tracker = BYTETracker(args, frame_rate=info["frame_rate"])
    result_rows = []
    input_count = 0

    for frame in range(1, info["frames"] + 1):
        frame_detections = detections[frame]
        input_count += len(frame_detections)
        if frame_detections:
            outputs = np.asarray(frame_detections, dtype=np.float32)
        else:
            outputs = np.empty((0, 5), dtype=np.float32)
        online_targets = tracker.update(
            outputs,
            (info["height"], info["width"]),
            (info["height"], info["width"]),
        )
        for target in online_targets:
            x, y, width, height = target.tlwh
            is_vertical = width / height > args.aspect_ratio_thresh
            if width * height > args.min_box_area and not is_vertical:
                result_rows.append((frame, target.track_id, x, y, width, height, target.score))

    write_sequence_results(output_dir / f"{info['name']}.txt", result_rows)
    return {
        "input_detection_rows": input_count,
        "output_tracking_rows": len(result_rows),
        "sequence": info["name"],
        "sequence_length": info["frames"],
    }


def main():
    args = parse_args()
    split_dir = args.dataset_root / args.split
    sequence_dirs = sorted(path for path in split_dir.glob("UAVSwarm-*") if path.is_dir())
    if not sequence_dirs:
        raise FileNotFoundError(f"no UAVSwarm sequences found under {split_dir}")
    if args.sequences is not None:
        requested = set(args.sequences)
        available = {path.name for path in sequence_dirs}
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError(f"unknown {args.split} sequences: {', '.join(unknown)}")
        sequence_dirs = [path for path in sequence_dirs if path.name in requested]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summaries = [track_sequence(sequence_dir, args.output_dir, args) for sequence_dir in sequence_dirs]
    summary = {
        "parameters": {
            "track_thresh": args.track_thresh,
            "det_thresh": args.det_thresh if args.det_thresh is not None else args.track_thresh + 0.1,
            "track_buffer": args.track_buffer,
            "match_thresh": args.match_thresh,
            "min_box_area": args.min_box_area,
            "aspect_ratio_thresh": args.aspect_ratio_thresh,
            "mot20": args.mot20,
            "affinity_checkpoint": str(args.affinity_checkpoint) if args.affinity_checkpoint else None,
            "affinity_device": args.affinity_device if args.affinity_checkpoint else None,
            "affinity_weight": args.affinity_weight if args.affinity_checkpoint else None,
            "affinity_min_probability": args.affinity_min_probability if args.affinity_checkpoint else None,
            "affinity_apply_to": args.affinity_apply_to if args.affinity_checkpoint else None,
        },
        "detections_root": str(args.detections_root) if args.detections_root else None,
        "sequences": summaries,
        "totals": {
            "input_detection_rows": sum(item["input_detection_rows"] for item in summaries),
            "output_tracking_rows": sum(item["output_tracking_rows"] for item in summaries),
            "sequences": len(summaries),
        },
    }
    with (args.output_dir / "tracking_summary.json").open("w") as destination:
        json.dump(summary, destination, indent=2, sort_keys=True)
        destination.write("\n")


if __name__ == "__main__":
    main()
