"""Append image-only proposals to a frozen cache without rewriting original rows.

Raw proposal scores select candidates; a fixed configured score is used only for
new tracker-input rows. The source proposal cache remains unchanged, and every
added raw score is retained in added.jsonl. Ground truth is never read.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from prebirth.infer import iou_xyxy, sha256, write_json


def parse_rows(content, label):
    rows = []
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = [float(value) for value in line.split(",")]
        except ValueError as error:
            raise ValueError(f"Invalid detection at {label}:{line_number}") from error
        if (len(values) < 7 or not np.isfinite(values).all()
                or values[0] != int(values[0]) or values[0] < 1
                or values[4] <= 0 or values[5] <= 0):
            raise ValueError(f"Invalid detection at {label}:{line_number}")
        rows.append((line_number, values))
    return rows


def as_xyxy(row):
    x, y, w, h = row[2:6]
    return [x, y, x + w, y + h]


def select_additions(original_rows, proposed_rows, width, height, first_frame, last_frame,
                     threshold, base_keep_score, overlap_iou, max_additions):
    """Pure geometry/score filter. Original rows are inspected but never modified."""
    base_by_frame = defaultdict(list)
    for _, row in original_rows:
        if row[6] >= base_keep_score:
            base_by_frame[int(row[0])].append(as_xyxy(row))
    proposed_by_frame = defaultdict(list)
    counts = {"raw_proposals": len(proposed_rows), "below_threshold": 0,
              "overlap_original": 0, "overlap_added": 0, "frame_budget": 0, "added": 0}
    for source_line, row in proposed_rows:
        frame = int(row[0])
        box = as_xyxy(row)
        if (not first_frame <= frame <= last_frame or not 0 <= row[6] <= 1
                or box[0] < 0 or box[1] < 0 or box[2] > width + 1e-4
                or box[3] > height + 1e-4):
            raise ValueError(f"Proposal outside frame/image bounds at source line {source_line}")
        if row[6] < threshold:
            counts["below_threshold"] += 1
            continue
        proposed_by_frame[frame].append((source_line, row, np.asarray(box, dtype=float)))
    additions = []
    for frame in sorted(proposed_by_frame):
        original = np.asarray(base_by_frame[frame], dtype=float).reshape(-1, 4)
        accepted = []
        candidates = sorted(proposed_by_frame[frame], key=lambda item: (-item[1][6], item[0]))
        for source_line, row, box in candidates:
            if np.any(iou_xyxy(box, original) >= overlap_iou):
                counts["overlap_original"] += 1
                continue
            if np.any(iou_xyxy(box, np.asarray(accepted).reshape(-1, 4)) >= overlap_iou):
                counts["overlap_added"] += 1
                continue
            if len(accepted) >= max_additions:
                counts["frame_budget"] += 1
                continue
            accepted.append(box)
            additions.append((source_line, row))
    counts["added"] = len(additions)
    return additions, counts


def merge_sequence(base_file, proposal_file, out, info, args):
    base_bytes = base_file.read_bytes()
    proposal_bytes = proposal_file.read_bytes()
    original = parse_rows(base_bytes, base_file)
    proposed = parse_rows(proposal_bytes, proposal_file)
    if info["frame_numbering"] == "original":
        first, last = info["source_frame_start"], info["source_frame_stop"]
    else:
        first, last = 1, info["source_frame_stop"] - info["source_frame_start"] + 1
    if any(not first <= int(row[0]) <= last for _, row in original):
        raise ValueError(f"Original/proposal frame numbering mismatch: {base_file}")
    additions, counts = select_additions(original, proposed, info["width"], info["height"],
        first, last, args.threshold, args.base_keep_score, args.overlap_iou, args.max_additions)
    out.mkdir(parents=True, exist_ok=False)
    added_bytes = bytearray()
    with (out / "added.jsonl").open("w") as log:
        for source_line, row in additions:
            values = [int(row[0]), -1, *row[2:6], args.score, -1, -1, -1]
            added_bytes.extend((",".join(str(v) for v in values) + "\n").encode())
            log.write(json.dumps({"frame": int(row[0]), "box_xywh": row[2:6],
                                  "raw_score": row[6], "tracker_score": args.score,
                                  "proposal_source_line": source_line}, allow_nan=False) + "\n")
    # Preserve even whitespace/precision/ordering of every original row. A single
    # delimiter newline is needed if the original file omitted its final newline.
    separator = b"\n" if base_bytes and not base_bytes.endswith(b"\n") and added_bytes else b""
    merged = base_bytes + separator + added_bytes
    target = out / "det.txt"
    target.write_bytes(merged)
    if target.read_bytes()[:len(base_bytes)] != base_bytes:
        raise AssertionError("Original cache byte prefix changed")
    return {"original_rows": len(original), **counts,
            "original_sha256": hashlib.sha256(base_bytes).hexdigest(),
            "proposal_sha256": hashlib.sha256(proposal_bytes).hexdigest(),
            "merged_sha256": sha256(target), "added_scores_sha256": sha256(out / "added.jsonl"),
            "original_bytes_preserved": True, "original_byte_length": len(base_bytes),
            "separator_newline_added": bool(separator)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-cache", type=Path, required=True)
    ap.add_argument("--proposals", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, required=True, help="Minimum raw proposal score")
    ap.add_argument("--score", type=float, default=0.75, help="Fixed score of new tracker-input rows")
    ap.add_argument("--base-keep-score", type=float, default=0.1)
    ap.add_argument("--overlap-iou", type=float, default=0.3)
    ap.add_argument("--max-additions", type=int, default=100, help="Per frame, after suppression")
    ap.add_argument("--sequences", nargs="+")
    args = ap.parse_args()
    if (not 0 <= args.threshold <= 1 or not 0 < args.score <= 1
            or not 0 <= args.base_keep_score <= 1 or not 0 < args.overlap_iou <= 1
            or args.max_additions <= 0):
        raise ValueError("Invalid threshold, overlap or candidate budget")
    source_metadata = json.loads((args.proposals / "metadata.json").read_text())
    if source_metadata.get("status") != "completed" or source_metadata.get("annotation_access") is not False:
        raise ValueError("Only completed, image-only proposal exports can be merged")
    names = sorted(source_metadata["sequences"])
    if args.sequences:
        requested = {name for item in args.sequences for name in item.split(",")}
        if requested - set(names):
            raise ValueError(f"Requested sequences missing from export: {sorted(requested - set(names))}")
        names = [name for name in names if name in requested]
    if not names:
        raise ValueError("No sequences to merge")
    for name in names:
        if not (args.base_cache / name / "det.txt").is_file():
            raise FileNotFoundError(args.base_cache / name / "det.txt")
        if sha256(args.proposals / name / "det.txt") != source_metadata["sequences"][name]["det_sha256"]:
            raise ValueError(f"Proposal digest mismatch: {name}")
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = {"status": "running", "command": sys.argv,
                "parameters": {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "source_metadata_sha256": sha256(args.proposals / "metadata.json"),
                "merge_source_sha256": sha256(__file__), "annotation_access": False,
                "original_policy": "Exact original file bytes retained as output prefix; new rows appended",
                "score_policy": "Raw proposal threshold; fixed tracker score for added rows only",
                "sequences": {}}
    write_json(args.out / "metadata.json", metadata)
    try:
        for name in names:
            result = merge_sequence(args.base_cache / name / "det.txt", args.proposals / name / "det.txt",
                                    args.out / name, source_metadata["sequences"][name], args)
            metadata["sequences"][name] = result
            write_json(args.out / "metadata.json", metadata)
            print(name, json.dumps(result), flush=True)
        metadata["status"] = "completed"
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(args.out / "metadata.json", metadata)


if __name__ == "__main__":
    main()
