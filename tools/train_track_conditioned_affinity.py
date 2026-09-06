#!/usr/bin/env python3
"""Fit TrackConditionedAffinity on UAVSwarmV2 MOT-train only."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from causal_lineage import sha256
from track_conditioned_affinity import extract_track_affinity_supervision, fit_track_affinity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--detections-root", type=Path, required=True)
    parser.add_argument("--fit-sequences", nargs="+", required=True)
    parser.add_argument("--validation-sequences", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--pair-top-k", type=int, default=5)
    parser.add_argument("--track-thresh", type=float, default=0.6)
    parser.add_argument("--det-thresh", type=float, default=0.7)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--match-thresh", type=float, default=0.9)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite output directory: {}".format(args.output_dir))
    if set(args.fit_sequences) & set(args.validation_sequences):
        raise ValueError("fit and validation sequences must be disjoint")
    if min(args.epochs, args.hidden_dim, args.batch_size, args.pair_top_k, args.track_buffer) < 1 or args.learning_rate <= 0.0:
        raise ValueError("epochs, hidden_dim, batch_size, pair_top_k, track_buffer and learning_rate must be positive")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0,1)")
    arguments = {
        "top_k": args.pair_top_k,
        "track_thresh": args.track_thresh,
        "det_thresh": args.det_thresh,
        "track_buffer": args.track_buffer,
        "match_thresh": args.match_thresh,
    }
    args.output_dir.mkdir(parents=True)
    fit, fit_summary = extract_track_affinity_supervision(args.dataset_root, args.detections_root, args.fit_sequences, **arguments)
    validation, validation_summary = extract_track_affinity_supervision(args.dataset_root, args.detections_root, args.validation_sequences, **arguments)
    outcome = fit_track_affinity(fit, validation, args.output_dir, args.device, args.seed, args.epochs, args.hidden_dim, args.dropout, args.learning_rate, args.batch_size)
    checkpoint = Path(outcome["checkpoint"])
    summary = {
        "schema_version": 1,
        "purpose": "GT-supervised track-state/detection same-identity affinity training.",
        "protocol_boundary": "MOT-train GT supplies identity labels only; inference consumes existing tracker state and detector boxes without GT or future frames.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "seed": args.seed,
        "parameters": {
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            **arguments,
        },
        "fit": fit_summary,
        "validation": validation_summary,
        "training": outcome,
        "checkpoint_sha256": sha256(checkpoint),
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(checkpoint), "checkpoint_sha256": summary["checkpoint_sha256"], "best": outcome["best"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
