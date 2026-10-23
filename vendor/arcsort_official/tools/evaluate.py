"""Wrapper around TrackEval for running MOT evaluation metrics.

Runs HOTA, CLEAR, and Identity metrics on tracking results using the
TrackEval toolkit via its Python API (not CLI, which has SEQMAP_FILE
parsing issues). Supports DanceTrack, MOT17, MOT20, and SportsMOT.

Usage:
    PYTHONPATH=. python tools/evaluate.py --benchmark dancetrack --split val \
        --gt data/DanceTrack/val \
        --trackers results/baseline_ocsort/

Requires TrackEval to be cloned at external/TrackEval/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from arcsort.utils import setup_logging


TRACKEVAL_DIR = Path(__file__).parent.parent / "external" / "TrackEval"

# Benchmark-specific settings
SEQMAP_DIR = Path(__file__).parent.parent / "data" / "seqmaps"

BENCHMARK_CONFIGS = {
    "dancetrack": {
        "seqmap_suffix": "_seqmap.txt",
        "seqmap_dir": None,
    },
    "mot17": {
        "seqmap_suffix": None,
        "seqmap_dir": SEQMAP_DIR,
    },
    "mot20": {
        "seqmap_suffix": None,
        "seqmap_dir": SEQMAP_DIR,
    },
    "sportsmot": {
        "seqmap_suffix": "_seqmap.txt",
        "seqmap_dir": None,
    },
    # SoccerNet-Tracking labels every object with class -1, so the MOTChallenge
    # pedestrian preprocessing would drop all GT; the official SoccerNet
    # evaluation also runs with preprocessing off
    "soccernet": {
        "seqmap_suffix": "_seqmap.txt",
        "seqmap_dir": None,
        "do_preproc": False,
    },
}


def run_evaluation(
    benchmark: str,
    split: str,
    gt_folder: Path,
    trackers_folder: Path,
    tracker_name: str = "",
    num_cores: int = 8,
) -> None:
    """Run TrackEval via Python API.

    Args:
        benchmark: Benchmark name (dancetrack, mot17, mot20, sportsmot, soccernet).
        split: Dataset split (train, val, test).
        gt_folder: Path to ground truth data.
        trackers_folder: Path to tracker results.
        tracker_name: Name of tracker subfolder. Empty for flat layout.
        num_cores: Number of parallel evaluation cores.

    Raises:
        ValueError: If benchmark is not supported.
        FileNotFoundError: If TrackEval is not installed.
    """
    if benchmark not in BENCHMARK_CONFIGS:
        raise ValueError(
            f"Unknown benchmark '{benchmark}'. "
            f"Supported: {list(BENCHMARK_CONFIGS.keys())}"
        )

    trackeval_path = str(TRACKEVAL_DIR)
    if trackeval_path not in sys.path:
        sys.path.insert(0, trackeval_path)

    import trackeval

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["USE_PARALLEL"] = True
    eval_config["NUM_PARALLEL_CORES"] = num_cores
    eval_config["PRINT_ONLY_COMBINED"] = True
    eval_config["PLOT_CURVES"] = False

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = str(gt_folder)
    dataset_config["TRACKERS_FOLDER"] = str(trackers_folder)
    dataset_config["TRACKERS_TO_EVAL"] = [tracker_name]
    dataset_config["TRACKER_SUB_FOLDER"] = ""
    dataset_config["SPLIT_TO_EVAL"] = split
    dataset_config["SKIP_SPLIT_FOL"] = True

    # Add seqmap if available
    cfg = BENCHMARK_CONFIGS[benchmark]
    dataset_config["DO_PREPROC"] = cfg.get("do_preproc", True)
    if cfg["seqmap_suffix"]:
        # DanceTrack/SportsMOT: seqmap next to GT folder (e.g., val_seqmap.txt)
        seqmap = gt_folder.parent / f"{split}{cfg['seqmap_suffix']}"
        if seqmap.exists():
            dataset_config["SEQMAP_FILE"] = str(seqmap)
    elif cfg.get("seqmap_dir"):
        # MOT17/MOT20: seqmap in data/seqmaps/ (e.g., MOT17-train.txt)
        seqmap_name = f"{benchmark.upper()}-{split}.txt"
        seqmap = cfg["seqmap_dir"] / seqmap_name
        if seqmap.exists():
            dataset_config["SEQMAP_FILE"] = str(seqmap)
        else:
            logger.warning(f"Seqmap not found: {seqmap}")

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity(),
    ]

    evaluator.evaluate(dataset_list, metrics_list)


def main() -> None:
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate tracking results with TrackEval")
    parser.add_argument(
        "--benchmark", type=str, required=True,
        choices=list(BENCHMARK_CONFIGS.keys()),
        help="Benchmark to evaluate on",
    )
    parser.add_argument(
        "--split", type=str, default="val",
        help="Dataset split (default: val)",
    )
    parser.add_argument(
        "--gt", type=str, required=True,
        help="Path to ground truth data folder",
    )
    parser.add_argument(
        "--trackers", type=str, required=True,
        help="Path to tracker results folder",
    )
    parser.add_argument(
        "--tracker-name", type=str, default="",
        help="Name of tracker subfolder (empty for flat layout)",
    )
    parser.add_argument(
        "--num-cores", type=int, default=8,
        help="Number of parallel evaluation cores (default: 8)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    gt_folder = Path(args.gt)
    trackers_folder = Path(args.trackers)

    if not gt_folder.exists():
        raise FileNotFoundError(f"Ground truth folder not found: {gt_folder}")
    if not trackers_folder.exists():
        raise FileNotFoundError(f"Tracker results folder not found: {trackers_folder}")

    logger.info(f"Running TrackEval for {args.benchmark} ({args.split} split)")

    run_evaluation(
        benchmark=args.benchmark,
        split=args.split,
        gt_folder=gt_folder,
        trackers_folder=trackers_folder,
        tracker_name=args.tracker_name,
        num_cores=args.num_cores,
    )

    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
