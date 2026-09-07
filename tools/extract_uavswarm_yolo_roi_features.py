#!/usr/bin/env python3
"""Create a GT-free frozen YOLO11s P3 ROI feature cache for scored detections."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from uavswarm_yolo_roi import (
    FrozenYOLO11sROIExtractor,
    extract_sequence_record,
    load_detections,
    read_sequence_info,
    save_sequence_record,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--detections-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ultralytics-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--feature-layer", type=int, default=16)
    parser.add_argument("--feature-stride", type=int, default=8)
    parser.add_argument("--feature-channels", type=int, default=128)
    parser.add_argument("--roi-output-size", type=int, default=3)
    parser.add_argument("--sequences", nargs="+", metavar="SEQUENCE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    split_root = args.dataset_root / args.split
    available = [path for path in sorted(split_root.glob("UAVSwarm-*")) if path.is_dir()]
    available_names = {path.name for path in available}
    selected_names = list(dict.fromkeys(args.sequences)) if args.sequences else [path.name for path in available]
    unknown = sorted(set(selected_names) - available_names)
    if unknown:
        raise ValueError(f"unknown {args.split} sequences: {', '.join(unknown)}")
    if args.image_size <= 0 or args.roi_output_size <= 0:
        raise ValueError("image-size and roi-output-size must be positive")

    args.output_dir.mkdir(parents=True)
    extractor = FrozenYOLO11sROIExtractor(
        checkpoint=args.checkpoint,
        ultralytics_root=args.ultralytics_root,
        feature_layer=args.feature_layer,
        feature_stride=args.feature_stride,
        expected_channels=args.feature_channels,
        image_size=args.image_size,
        roi_output_size=args.roi_output_size,
        device_name=args.device,
    )
    sequences = {}
    try:
        for name in selected_names:
            sequence_dir = split_root / name
            info = read_sequence_info(sequence_dir)
            det_path = args.detections_root / name / "det.txt"
            detections = load_detections(det_path)
            record = extract_sequence_record(sequence_dir, info, detections, extractor)
            record_path = args.output_dir / f"{name}.npz"
            save_sequence_record(record_path, record)
            sequences[name] = {
                "frames": int(info["frames"]),
                "detection_rows": int(len(record["frame_ids"])),
                "feature_dimension": int(record["features"].shape[1]),
                "det_path": str(det_path.resolve()),
                "det_sha256": sha256(det_path),
                "seqinfo_sha256": sha256(sequence_dir / "seqinfo.ini"),
                "record": record_path.name,
                "record_sha256": sha256(record_path),
            }
            print(json.dumps({"sequence": name, **sequences[name]}, ensure_ascii=False), flush=True)
    finally:
        extractor.close()

    manifest = {
        "schema_version": 1,
        "purpose": "Frozen YOLO11s P3 ROI features aligned to an existing scored detector cache.",
        "protocol_boundary": "Reads detector cache, images, frozen checkpoint, and seqinfo only. GT is not read.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"root": str(args.dataset_root.resolve()), "split": args.split},
        "detector_cache_root": str(args.detections_root.resolve()),
        "model": {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint)},
        "feature": {
            "layer": args.feature_layer,
            "stride": args.feature_stride,
            "channels": args.feature_channels,
            "image_size": args.image_size,
            "roi_output_size": args.roi_output_size,
            "pooling": "ROIAlign aligned=True then spatial mean and L2 normalization",
            "device": args.device,
        },
        "sequences": sequences,
        "totals": {
            "sequences": len(sequences),
            "frames": sum(row["frames"] for row in sequences.values()),
            "detection_rows": sum(row["detection_rows"] for row in sequences.values()),
        },
    }
    (args.output_dir / "feature_cache_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), **manifest["totals"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
