"""Run tracker on pre-cached detection files.

Reads cached detection results (one .txt file per sequence, MOT format)
and runs the ARC-SORT tracker. Outputs tracking results in MOT format
for evaluation with TrackEval.

Detection cache format (per line):
    frame_id, -1, x1, y1, w, h, confidence, -1, -1, -1

Output format (per line):
    frame_id, track_id, x1, y1, w, h, confidence, -1, -1, -1

Usage:
    # Cached CMC + cached embeddings (fast, CPU only; see scripts/reproduce.sh)
    python tools/run_cached.py --config arcsort/configs/sportsmot.yaml \
        --detections cache/detections/sportsmot_val/ \
        --cmc-cache cache/cmc/sportsmot_val/ \
        --emb-cache cache/embeddings/sportsmot_val/ \
        --output outputs/sportsmot_val/online/

    # Compute CMC from frames instead of a cache:
    python tools/run_cached.py --config arcsort/configs/dancetrack.yaml \
        --detections cache/detections/dancetrack_val/ \
        --frames data/DanceTrack/val \
        --emb-cache cache/embeddings/dancetrack_val/ \
        --output outputs/dancetrack_val/online/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from arcsort.tracker import ARCTracker, load_config
from arcsort.utils import setup_logging


def load_detections(
    det_file: Path,
    min_confidence: float = 0.0,
) -> dict[int, np.ndarray]:
    """Load cached detections from a MOT-format text file.

    Args:
        det_file: Path to detection file. Each line contains:
            frame_id, -1, x1, y1, w, h, confidence, -1, -1, -1
        min_confidence: Discard detections below this confidence during
            loading to reduce memory and I/O cost. Set to the tracker's
            track_low_thresh (or slightly below) for efficiency.

    Returns:
        Dict mapping frame_id to detection array of shape (N, 6)
        with columns [x1, y1, x2, y2, confidence, class_id].

    Raises:
        FileNotFoundError: If detection file does not exist.
    """
    if not det_file.exists():
        raise FileNotFoundError(f"Detection file not found: {det_file}")

    data = np.loadtxt(det_file, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[0] == 0:
        return {}

    # Pre-filter by confidence to avoid processing millions of junk detections
    if min_confidence > 0:
        conf_mask = data[:, 6] >= min_confidence
        data = data[conf_mask]
        if data.shape[0] == 0:
            return {}

    detections_by_frame: dict[int, np.ndarray] = {}

    frame_ids = data[:, 0].astype(int)
    unique_frames = np.unique(frame_ids)

    for frame_id in unique_frames:
        mask = frame_ids == frame_id
        frame_data = data[mask]

        # Convert from [frame, id, x, y, w, h, conf, ...] to [x1, y1, x2, y2, conf, cls]
        x1 = frame_data[:, 2]
        y1 = frame_data[:, 3]
        w = frame_data[:, 4]
        h = frame_data[:, 5]
        conf = frame_data[:, 6]

        dets = np.column_stack([x1, y1, x1 + w, y1 + h, conf, np.zeros(len(conf))])
        detections_by_frame[frame_id] = dets

    return detections_by_frame


def run_sequence(
    tracker: ARCTracker,
    detections_by_frame: dict[int, np.ndarray],
    frames_dir: Path | None = None,
    cmc_cache: dict[int, np.ndarray] | None = None,
) -> list[str]:
    """Run tracker on a single sequence.

    Args:
        tracker: Initialized ARCTracker instance.
        detections_by_frame: Dict mapping frame_id to detections.
        frames_dir: Path to sequence frame images (e.g.,
            data/DanceTrack/val/dancetrack0004/img1/). Required when
            tracker has CMC enabled and no cmc_cache. Images named %08d.jpg.
        cmc_cache: Pre-computed CMC affine transforms as dict mapping
            frame_id to 2x3 affine matrix. When provided, frames_dir
            is not needed for CMC (but may still be needed for ReID).

    Returns:
        List of output lines in MOT format.
    """
    tracker.reset()

    if not detections_by_frame:
        return []

    max_frame = max(detections_by_frame.keys())
    output_lines = []
    use_frames = frames_dir is not None
    use_cmc_cache = cmc_cache is not None

    # Determine if we need to load frames (for CMC and/or ReID).
    # ReID served from an embedding cache needs no frames.
    use_reid = (
        tracker.cfg.get("use_reid", False)
        or tracker.cfg.get("use_gated_reid", False)
    ) and not (
        tracker.embedding_computer is not None
        and tracker.embedding_computer.has_cache()
    )

    for frame_id in range(1, max_frame + 1):
        dets = detections_by_frame.get(frame_id, np.empty((0, 6)))

        frame = None
        cmc_affine = None

        if use_cmc_cache:
            cmc_affine = cmc_cache.get(frame_id)

        # Load frame if needed for CMC (without cache) or ReID
        if use_frames and (use_reid or (not use_cmc_cache)):
            # Try common frame naming conventions (8-digit for DanceTrack,
            # 6-digit for MOT17/MOT20)
            for fmt in (f"{frame_id:08d}.jpg", f"{frame_id:06d}.jpg"):
                frame_path = frames_dir / fmt
                if frame_path.exists():
                    frame = cv2.imread(str(frame_path))
                    break

        tracked = tracker.update(dets, frame=frame, cmc_affine=cmc_affine)

        for row in tracked:
            x1, y1, x2, y2, track_id = row
            w = x2 - x1
            h = y2 - y1
            line = f"{frame_id},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1"
            output_lines.append(line)

    return output_lines


def main() -> None:
    """Main entry point for cached tracking."""
    parser = argparse.ArgumentParser(description="Run ARC-SORT on cached detections")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to tracker config YAML file",
    )
    parser.add_argument(
        "--detections", type=str, required=True,
        help="Directory containing cached detection .txt files (one per sequence)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Directory to write tracking results",
    )
    parser.add_argument(
        "--frames", type=str, default=None,
        help="Directory containing sequence frame images (required for CMC "
             "if --cmc-cache is not provided). "
             "Expected structure: <frames>/<seq_name>/img1/<frame_id>.jpg",
    )
    parser.add_argument(
        "--cmc-cache", type=str, default=None,
        help="Directory containing pre-computed CMC .npy files (one per "
             "sequence). Generated by tools/cache_cmc.py. When provided, "
             "skips frame loading for CMC (much faster).",
    )
    parser.add_argument(
        "--emb-cache", type=str, default=None,
        help="Directory containing pre-computed ReID embedding .npz files "
             "(one per sequence). Generated by tools/cache_embeddings.py. "
             "When provided, ReID runs from the cache: no GPU and no frame "
             "loading needed.",
    )
    parser.add_argument(
        "--output-box", choices=("detection", "kalman"), default=None,
        help="Override the config's output_box. Use 'kalman' to reproduce "
             "the paper results.",
    )
    parser.add_argument(
        "--mot17-triplicate", action="store_true",
        help="After tracking, copy each MOT17-XX.txt result to "
             "MOT17-XX-DPM.txt, MOT17-XX-FRCNN.txt, MOT17-XX-SDP.txt "
             "(required for MOT17 submission which expects all 3 variants).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    config = load_config(args.config)
    if args.output_box:
        config["output_box"] = args.output_box
    logger.info(f"Loaded config from {args.config}")
    logger.debug(f"Config: {config}")

    det_dir = Path(args.detections)
    if not det_dir.exists():
        raise FileNotFoundError(f"Detection directory not found: {det_dir}")

    frames_base = Path(args.frames) if args.frames else None
    cmc_cache_dir = Path(args.cmc_cache) if args.cmc_cache else None
    emb_cache_dir = Path(args.emb_cache) if args.emb_cache else None
    use_cmc = config.get("use_cmc", False)
    use_reid = config.get("use_reid", False) or config.get("use_gated_reid", False)

    if use_cmc and frames_base is None and cmc_cache_dir is None:
        raise ValueError(
            "CMC is enabled in config but neither --frames nor --cmc-cache "
            "was provided. CMC requires frame images or cached transforms."
        )
    if use_reid and frames_base is None and emb_cache_dir is None:
        raise ValueError(
            "ReID is enabled in config but neither --frames nor --emb-cache "
            "was provided. ReID requires frame images for crop extraction "
            "or pre-computed embeddings."
        )
    if frames_base is not None and not frames_base.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_base}")
    if cmc_cache_dir is not None and not cmc_cache_dir.exists():
        raise FileNotFoundError(f"CMC cache directory not found: {cmc_cache_dir}")
    if emb_cache_dir is not None and not emb_cache_dir.exists():
        raise FileNotFoundError(f"Embedding cache directory not found: {emb_cache_dir}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find detection files
    det_files = sorted(det_dir.glob("*.txt"))
    if not det_files:
        raise FileNotFoundError(f"No .txt detection files found in {det_dir}")

    logger.info(f"Found {len(det_files)} sequences in {det_dir}")

    tracker = ARCTracker(config)

    for det_file in det_files:
        seq_name = det_file.stem
        logger.info(f"Processing sequence: {seq_name}")

        # Load per-sequence embedding cache into the tracker's computer
        seq_emb_loaded = False
        if emb_cache_dir is not None and tracker.embedding_computer is not None:
            emb_file = emb_cache_dir / f"{seq_name}.npz"
            if not emb_file.exists():
                # MOT17 fallback naming (MOT17-02 vs MOT17-02-SDP)
                for suffix in ["SDP", "FRCNN", "DPM"]:
                    alt = emb_cache_dir / f"{seq_name}-{suffix}.npz"
                    if alt.exists():
                        emb_file = alt
                        break
            if emb_file.exists():
                tracker.embedding_computer.load_cache(emb_file)
                seq_emb_loaded = True
            else:
                logger.warning(
                    f"Embedding cache not found for {seq_name}; "
                    "falling back to frames (if provided)"
                )

        # Resolve CMC source: cached transforms or raw frames
        seq_cmc_cache = None
        seq_frames_dir = None

        if cmc_cache_dir is not None:
            cmc_file = cmc_cache_dir / f"{seq_name}.npy"
            if cmc_file.exists():
                seq_cmc_cache = np.load(str(cmc_file), allow_pickle=True).item()
            else:
                # MOT17 fallback: detection is MOT17-02.txt but CMC is MOT17-02-SDP.npy
                for suffix in ["SDP", "FRCNN", "DPM"]:
                    alt = cmc_cache_dir / f"{seq_name}-{suffix}.npy"
                    if alt.exists():
                        seq_cmc_cache = np.load(str(alt), allow_pickle=True).item()
                        break
                else:
                    logger.warning(
                        f"CMC cache not found for {seq_name}: {cmc_file}"
                    )

        # Load frame paths if needed for CMC (without cache) or ReID
        # (without embedding cache)
        needs_reid_frames = use_reid and not seq_emb_loaded
        if (seq_cmc_cache is None or needs_reid_frames) and frames_base is not None:
            seq_frames_dir = frames_base / seq_name / "img1"
            if not seq_frames_dir.exists():
                # MOT17 fallback: detection file is MOT17-02.txt but frames
                # are in MOT17-02-SDP/img1/ (all variants share same images)
                for suffix in ["SDP", "FRCNN", "DPM"]:
                    alt = frames_base / f"{seq_name}-{suffix}" / "img1"
                    if alt.exists():
                        seq_frames_dir = alt
                        break
                else:
                    logger.warning(
                        f"Frame directory not found for {seq_name}: "
                        f"{seq_frames_dir}, running without CMC for this sequence"
                    )
                    seq_frames_dir = None

        # Pre-filter detections below track_low_thresh during loading
        # to avoid processing millions of low-confidence false positives
        min_conf = config.get("track_low_thresh", 0.1)
        detections_by_frame = load_detections(det_file, min_confidence=min_conf)
        output_lines = run_sequence(
            tracker,
            detections_by_frame,
            frames_dir=seq_frames_dir,
            cmc_cache=seq_cmc_cache,
        )

        out_file = out_dir / f"{seq_name}.txt"
        with open(out_file, "w") as f:
            f.write("\n".join(output_lines))
            if output_lines:
                f.write("\n")

        # Ambiguity / arbitration sidecars (.json so TrackEval and the
        # offline tools, which glob *.txt, never see them)
        amb_events = tracker.pop_ambiguity_events()
        if amb_events:
            import json
            with open(out_dir / f"{seq_name}.ambiguity.json", "w") as f:
                json.dump(amb_events, f, indent=1)
        arb_swaps = getattr(tracker, "arbitration_swaps", [])
        if arb_swaps:
            import json
            with open(out_dir / f"{seq_name}.swaps.json", "w") as f:
                json.dump(arb_swaps, f, indent=1)

        num_frames = len(detections_by_frame)
        num_tracks = len(set(
            int(line.split(",")[1]) for line in output_lines
        )) if output_lines else 0
        logger.info(f"  {num_frames} frames, {num_tracks} unique tracks")

    # MOT17 triplicate: copy MOT17-XX.txt -> MOT17-XX-{DPM,FRCNN,SDP}.txt
    if args.mot17_triplicate:
        import shutil

        triplicates = ["DPM", "FRCNN", "SDP"]
        result_files = sorted(out_dir.glob("MOT17-*.txt"))
        count = 0
        for result_file in result_files:
            base_name = result_file.stem
            # Skip files already in triplicate form (e.g., MOT17-01-SDP)
            if any(base_name.endswith(f"-{ext}") for ext in triplicates):
                continue
            for ext in triplicates:
                dest = out_dir / f"{base_name}-{ext}.txt"
                shutil.copy2(result_file, dest)
                count += 1
            # Remove the base file (submission only needs the 3 variants)
            result_file.unlink()
        logger.info(f"MOT17 triplicate: created {count} variant files")

    logger.info(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
