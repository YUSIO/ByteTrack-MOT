"""Extract frozen Seed features, fit peer heads, score, or filter saved A rows."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fctrack.peer_relation import extract, train, score, apply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    stages = {
        "extract": ("root", "config", "cell", "output", "device"),
        "train": ("extraction", "labels", "config", "output"),
        "score": ("extraction", "trained", "config", "output"),
        "apply": ("scoring", "thresholds", "output"),
    }
    for stage, fields in stages.items():
        command = commands.add_parser(stage)
        for field in fields:
            command.add_argument("--" + field, required=True)
    args = vars(parser.parse_args())
    stage = args.pop("stage")
    fn = {"extract": extract, "train": train, "score": score, "apply": apply}[stage]
    for key in ("config", "labels", "thresholds"):
        if key in args:
            args[key + "_path"] = args.pop(key)
    result = fn(**args)
    print(json.dumps({key: result[key] for key in ("status", "stage", "input_rows", "wall_seconds")}))


if __name__ == "__main__":
    main()
