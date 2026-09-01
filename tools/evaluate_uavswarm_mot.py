#!/usr/bin/env python3
"""Evaluate UAVSwarm MOT results with motmetrics and TrackEval's HOTA metric."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import motmetrics as mm
import numpy as np
from trackeval.metrics import HOTA

# motmetrics 1.4.0 still calls np.asfarray, which NumPy 2 removed.  The old
# function's default behavior is exactly np.asarray(..., dtype=float), so this
# local evaluator compatibility alias does not change any distance computation.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda values, dtype=float: np.asarray(values, dtype=dtype)


COUNT_FIELDS = {
    "num_unique_objects",
    "mostly_tracked",
    "partially_tracked",
    "mostly_lost",
    "num_false_positives",
    "num_misses",
    "num_switches",
    "num_fragmentations",
    "num_objects",
    "num_predictions",
}
RATE_FIELDS = {"idf1", "idp", "idr", "recall", "precision", "mota"}
MOT_FIELDS = [
    "idf1",
    "idp",
    "idr",
    "recall",
    "precision",
    "num_unique_objects",
    "mostly_tracked",
    "partially_tracked",
    "mostly_lost",
    "num_false_positives",
    "num_misses",
    "num_switches",
    "num_fragmentations",
    "mota",
    "motp",
    "num_objects",
    "num_predictions",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--tracker-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def read_mot_boxes(path, min_confidence):
    frames = defaultdict(list)
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            fields = line.strip().split(",")
            if len(fields) < 6:
                raise ValueError(f"{path}:{line_number}: expected at least 6 MOT fields")
            confidence = float(fields[6]) if len(fields) > 6 else 1.0
            if confidence < min_confidence:
                continue
            frame = int(float(fields[0]))
            track_id = int(float(fields[1]))
            x, y, width, height = (float(value) for value in fields[2:6])
            frames[frame].append((track_id, x, y, width, height))
    return frames


def iou_matrix(gt_boxes, tracker_boxes):
    if not gt_boxes or not tracker_boxes:
        return np.empty((len(gt_boxes), len(tracker_boxes)), dtype=float)
    gt = np.asarray([box[1:] for box in gt_boxes], dtype=float)
    tracker = np.asarray([box[1:] for box in tracker_boxes], dtype=float)
    gt_right = gt[:, 0] + gt[:, 2]
    gt_bottom = gt[:, 1] + gt[:, 3]
    tracker_right = tracker[:, 0] + tracker[:, 2]
    tracker_bottom = tracker[:, 1] + tracker[:, 3]
    left = np.maximum(gt[:, None, 0], tracker[None, :, 0])
    top = np.maximum(gt[:, None, 1], tracker[None, :, 1])
    right = np.minimum(gt_right[:, None], tracker_right[None, :])
    bottom = np.minimum(gt_bottom[:, None], tracker_bottom[None, :])
    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    union = gt[:, None, 2] * gt[:, None, 3] + tracker[None, :, 2] * tracker[None, :, 3] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def trackeval_data(gt_frames, tracker_frames):
    gt_source_ids = sorted({box[0] for boxes in gt_frames.values() for box in boxes})
    tracker_source_ids = sorted({box[0] for boxes in tracker_frames.values() for box in boxes})
    gt_id_map = {source_id: index for index, source_id in enumerate(gt_source_ids)}
    tracker_id_map = {source_id: index for index, source_id in enumerate(tracker_source_ids)}
    frame_ids = sorted(set(gt_frames) | set(tracker_frames))
    gt_ids, tracker_ids, similarities = [], [], []
    for frame in frame_ids:
        gt_boxes = gt_frames[frame]
        tracker_boxes = tracker_frames[frame]
        gt_ids.append(np.asarray([gt_id_map[box[0]] for box in gt_boxes], dtype=int))
        tracker_ids.append(np.asarray([tracker_id_map[box[0]] for box in tracker_boxes], dtype=int))
        similarities.append(iou_matrix(gt_boxes, tracker_boxes))
    return {
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": similarities,
        "num_gt_ids": len(gt_source_ids),
        "num_tracker_ids": len(tracker_source_ids),
        "num_gt_dets": sum(len(boxes) for boxes in gt_frames.values()),
        "num_tracker_dets": sum(len(boxes) for boxes in tracker_frames.values()),
    }


def serialise_mot_row(row):
    result = {}
    for field in MOT_FIELDS:
        value = row[field]
        if field in COUNT_FIELDS:
            result[field] = int(value)
        elif field in RATE_FIELDS:
            result[field] = round(100.0 * float(value), 6)
        else:
            result[field] = round(float(value), 6)
    return result


def serialise_hota_result(result, metric):
    summary = {
        field: round(100.0 * float(np.mean(result[field])), 6)
        for field in ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA", "OWTA")
    }
    summary["curve_percent"] = {
        field: [round(100.0 * float(value), 6) for value in result[field]]
        for field in ("HOTA", "DetA", "AssA")
    }
    summary["iou_thresholds"] = [round(float(value), 2) for value in metric.array_labels]
    return summary


def main():
    args = parse_args()
    sequence_dirs = sorted(path for path in (args.dataset_root / args.split).glob("UAVSwarm-*") if path.is_dir())
    if not sequence_dirs:
        raise FileNotFoundError(f"no UAVSwarm sequences found under {args.dataset_root / args.split}")
    mm.lap.default_solver = "lap"
    mot_accumulators, sequence_names, hota_by_sequence = [], [], {}
    for sequence_dir in sequence_dirs:
        sequence_name = sequence_dir.name
        gt_path = sequence_dir / "gt" / "gt.txt"
        tracker_path = args.tracker_results / f"{sequence_name}.txt"
        if not tracker_path.is_file():
            raise FileNotFoundError(f"missing tracker result {tracker_path}")
        gt_for_motmetrics = mm.io.loadtxt(str(gt_path), fmt="mot15-2D", min_confidence=1)
        tracker_for_motmetrics = mm.io.loadtxt(str(tracker_path), fmt="mot15-2D", min_confidence=-1)
        mot_accumulators.append(
            mm.utils.compare_to_groundtruth(gt_for_motmetrics, tracker_for_motmetrics, "iou", distth=args.iou_threshold)
        )
        sequence_names.append(sequence_name)
        hota_by_sequence[sequence_name] = HOTA().eval_sequence(
            trackeval_data(read_mot_boxes(gt_path, 1.0), read_mot_boxes(tracker_path, -1.0))
        )

    mot_summary = mm.metrics.create().compute_many(
        mot_accumulators, names=sequence_names, metrics=MOT_FIELDS, generate_overall=True
    )
    hota_metric = HOTA()
    hota_overall = hota_metric.combine_sequences(hota_by_sequence)
    per_sequence = {}
    for sequence_name in sequence_names:
        per_sequence[sequence_name] = {
            "motmetrics": serialise_mot_row(mot_summary.loc[sequence_name]),
            "hota": serialise_hota_result(hota_by_sequence[sequence_name], hota_metric),
        }
    output = {
        "units": {"rates": "percent", "counts": "events or detections", "motp": "IoU distance (lower is better)"},
        "protocol": {
            "clear_and_identity_iou_threshold": args.iou_threshold,
            "hota_implementation": "TrackEval 1.1.0 HOTA over IoU thresholds 0.05 through 0.95",
        },
        "overall": {
            "motmetrics": serialise_mot_row(mot_summary.loc["OVERALL"]),
            "hota": serialise_hota_result(hota_overall, hota_metric),
        },
        "per_sequence": per_sequence,
    }
    with args.output.open("w") as destination:
        json.dump(output, destination, indent=2, sort_keys=True)
        destination.write("\n")


if __name__ == "__main__":
    main()
