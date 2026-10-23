"""Post-processing: linear interpolation of tracking result gaps.

For each track ID, finds frames where the track has output both before and
after but no output during the gap, and linearly interpolates bounding boxes.
This recovers missed detections and directly boosts DetA/MOTA.

Standard post-processing used by nearly all competitive DanceTrack entries.

Input/output format (MOT):
    frame_id, track_id, x1, y1, w, h, confidence, -1, -1, -1

Usage:
    python tools/interpolate_tracks.py \
        --input results/arcsort_full/ \
        --output results/arcsort_full_interp/

    # Custom max gap (default 10 frames):
    python tools/interpolate_tracks.py \
        --input results/arcsort_full/ \
        --output results/arcsort_full_interp/ \
        --max-gap 15
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from loguru import logger


def interpolate_sequence(
    lines: list[str],
    max_gap: int = 10,
) -> list[str]:
    """Interpolate gaps in a single sequence's tracking output.

    Args:
        lines: MOT-format lines for one sequence.
        max_gap: Maximum gap length (in frames) to interpolate. Gaps
            longer than this are left unfilled to avoid hallucinating
            tracks through long occlusions.

    Returns:
        Augmented list of MOT-format lines with interpolated entries.
    """
    if not lines:
        return []

    # Parse: group by track_id -> list of (frame_id, x1, y1, w, h, conf)
    tracks: dict[int, list[tuple[int, float, float, float, float, float]]] = defaultdict(list)

    for line in lines:
        parts = line.strip().split(",")
        if len(parts) < 7:
            continue
        frame_id = int(parts[0])
        track_id = int(parts[1])
        x1 = float(parts[2])
        y1 = float(parts[3])
        w = float(parts[4])
        h = float(parts[5])
        conf = float(parts[6])
        tracks[track_id].append((frame_id, x1, y1, w, h, conf))

    # Sort each track's detections by frame
    for tid in tracks:
        tracks[tid].sort(key=lambda x: x[0])

    # Generate interpolated entries
    new_lines = list(lines)  # Start with originals
    interp_count = 0

    for tid, dets in tracks.items():
        for i in range(len(dets) - 1):
            f1, x1_1, y1_1, w1, h1, c1 = dets[i]
            f2, x1_2, y1_2, w2, h2, c2 = dets[i + 1]

            gap = f2 - f1
            if gap <= 1 or gap > max_gap:
                continue

            # Linear interpolation for each gap frame
            for f in range(f1 + 1, f2):
                alpha = (f - f1) / gap
                x1_i = x1_1 + alpha * (x1_2 - x1_1)
                y1_i = y1_1 + alpha * (y1_2 - y1_1)
                w_i = w1 + alpha * (w2 - w1)
                h_i = h1 + alpha * (h2 - h1)
                c_i = c1 + alpha * (c2 - c1)

                line = (
                    f"{f},{tid},{x1_i:.2f},{y1_i:.2f},"
                    f"{w_i:.2f},{h_i:.2f},{c_i:.4f},-1,-1,-1"
                )
                new_lines.append(line)
                interp_count += 1

    if interp_count > 0:
        logger.debug(f"  Interpolated {interp_count} entries")

    # Sort output by (frame_id, track_id) for clean MOT format
    def sort_key(line: str) -> tuple[int, int]:
        parts = line.strip().split(",")
        return (int(parts[0]), int(parts[1]))

    new_lines.sort(key=sort_key)
    return new_lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interpolate gaps in MOT tracking results"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Directory containing tracking result .txt files",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Directory to write interpolated results",
    )
    parser.add_argument(
        "--max-gap", type=int, default=10,
        help="Maximum gap length to interpolate (default: 10)",
    )
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)

    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Sequence results only -- skip TrackEval artifacts living alongside
    txt_files = [
        f for f in sorted(in_dir.glob("*.txt"))
        if not f.name.startswith("pedestrian_")
    ]
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {in_dir}")

    logger.info(f"Interpolating {len(txt_files)} sequences (max_gap={args.max_gap})")

    total_original = 0
    total_after = 0

    for txt_file in txt_files:
        with open(txt_file) as f:
            lines = [l.strip() for l in f if l.strip()]

        total_original += len(lines)
        interp_lines = interpolate_sequence(lines, max_gap=args.max_gap)
        total_after += len(interp_lines)

        out_file = out_dir / txt_file.name
        with open(out_file, "w") as f:
            f.write("\n".join(interp_lines))
            if interp_lines:
                f.write("\n")

        logger.info(
            f"  {txt_file.stem}: {len(lines)} -> {len(interp_lines)} "
            f"(+{len(interp_lines) - len(lines)} interpolated)"
        )

    logger.info(
        f"Total: {total_original} -> {total_after} "
        f"(+{total_after - total_original} interpolated)"
    )
    logger.info(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
