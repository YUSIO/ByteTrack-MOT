"""Filter frozen A output lines with original Seed p >= .97; no GT or tracking."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from fctrack.output_verification import verify_output_tracks  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    summary = verify_output_tracks(args.root, args.config, args.cell, args.output, args.device)
    print(json.dumps({key: summary[key] for key in (
        "status", "cell", "sequence_count", "frame_count", "scored_frame_count",
        "input_rows", "kept_rows", "rejected_rows", "wall_seconds")}), flush=True)


if __name__ == "__main__":
    main()
