#!/usr/bin/env python3
"""Apply a CLLR checkpoint without reading GT and emit a MOT detector cache."""

import argparse
import json
from pathlib import Path

import torch

from causal_lineage import rescore_cache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quality-threshold", type=float, required=True)
    parser.add_argument("--link-threshold", type=float, required=True)
    parser.add_argument("--history-minimum", type=int, default=2)
    parser.add_argument("--promotion-floor", type=float, default=0.71)
    parser.add_argument("--pair-top-k", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    result = rescore_cache(
        args.dataset_root, args.input_root, args.output_root, args.checkpoint,
        args.sequences, args.split, args.device, args.quality_threshold,
        args.link_threshold, args.history_minimum, args.promotion_floor, args.pair_top_k,
    )
    print(json.dumps(result["totals"], ensure_ascii=False))


if __name__ == "__main__":
    main()
