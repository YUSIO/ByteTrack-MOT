#!/usr/bin/env python3
"""Train CLLR from UAVSwarmV2 MOT-train GT and frozen detector outputs."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from causal_lineage import extract_supervision, fit_models, sha256


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
    parser.add_argument("--pair-top-k", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite output directory: {}".format(args.output_dir))
    if set(args.fit_sequences) & set(args.validation_sequences):
        raise ValueError("fit and validation sequences must be disjoint")
    if args.epochs < 1 or args.hidden_dim < 1 or args.batch_size < 1 or args.pair_top_k < 1:
        raise ValueError("epochs, hidden_dim, batch_size and pair_top_k must be positive")
    if not 0.0 <= args.dropout < 1.0 or args.learning_rate <= 0.0:
        raise ValueError("dropout must be in [0,1) and learning_rate positive")
    args.output_dir.mkdir(parents=True)
    fit, fit_summary = extract_supervision(args.dataset_root, args.detections_root, args.fit_sequences, top_k=args.pair_top_k)
    validation, validation_summary = extract_supervision(args.dataset_root, args.detections_root, args.validation_sequences, top_k=args.pair_top_k)
    outcome = fit_models(
        fit, validation, args.output_dir, args.device, args.seed, args.epochs,
        args.hidden_dim, args.dropout, args.learning_rate, args.batch_size,
    )
    checkpoint = Path(outcome["checkpoint"])
    summary = {
        "schema_version": 1,
        "purpose": "GT-supervised causal detector objectness and adjacent-frame lineage training.",
        "protocol_boundary": "MOT-train GT creates labels only; the serialized model accepts detector-only causal features.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "seed": args.seed,
        "parameters": {"epochs": args.epochs, "hidden_dim": args.hidden_dim, "dropout": args.dropout, "learning_rate": args.learning_rate, "batch_size": args.batch_size, "pair_top_k": args.pair_top_k},
        "fit": fit_summary,
        "validation": validation_summary,
        "training": outcome,
        "checkpoint_sha256": sha256(checkpoint),
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(checkpoint), "checkpoint_sha256": summary["checkpoint_sha256"], "best": outcome["best"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
