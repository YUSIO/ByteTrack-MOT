"""Extract fixed Seed spatial interventions or apply frozen output thresholds."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fctrack.spatial_intervention import apply, extract  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extraction = commands.add_parser("extract", help="GT-free six-view current-crop scoring")
    extraction.add_argument("--root", type=Path, required=True)
    extraction.add_argument("--config", type=Path, required=True)
    extraction.add_argument("--cell", required=True)
    extraction.add_argument("--output", type=Path, required=True)
    extraction.add_argument("--device", required=True)
    filtering = commands.add_parser("apply", help="exact source-line filtering; no inference or GT")
    filtering.add_argument("--extraction", type=Path, required=True)
    filtering.add_argument("--thresholds", type=Path, required=True)
    filtering.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "extract":
        summary = extract(args.root, args.config, args.cell, args.output, args.device)
    else:
        summary = apply(args.extraction, args.thresholds, args.output)
    print(json.dumps({k: summary[k] for k in ("status", "stage", "sequence_count", "frame_count",
                                              "input_rows", "wall_seconds")}), flush=True)


if __name__ == "__main__":
    main()
