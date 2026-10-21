"""Read-only capture of A_orig queries and pre-gate real detector anchors.

The controller wraps the tracker instance's update, returns its original result,
and copies detector observations before FCTrack can recover, birth or relabel.
The capture never opens GT. After capture and regression checks, a separate
canonical evaluator reads GT for metrics only. Frozen Exp139 event fields are
read only for regression checks; their GT labels are not observation features.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


SIGNATURE_FIELDS = ("frame", "candidate_index", "proposal", "final", "p", "native_accept")
CHECK_FIELDS = ("sequence", "kind", "frame", "candidate_index", "detection_index", "track_id",
                "proposal", "final", "p", "native_accept", "threshold", "gap", "since_real", "obs",
                "estimator", "det_score", "signature", "outcome", "decision_accept", "intervened",
                "track_id_after", "output_ids_after_relabel")


def signature(event):
    # Exactly the Exp139 action_diagnostic.signature definition, without importing its GT controller.
    fields = {key: event[key] for key in SIGNATURE_FIELDS}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_queries(path):
    # Select an explicit allowlist. GT-derived labels in the reference are never observation features.
    result = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        result.append({key: row[key] for key in CHECK_FIELDS if key in row})
    return result


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")


class ProbeController:
    def __init__(self, name, fc, seq_dir, destination, anchor_score=.6):
        require(fc.decision_hook is None, "A pre-existing decision hook must not be replaced")
        self.name, self.fc = name, fc
        self.seq_dir, self.destination = Path(seq_dir), Path(destination)
        self.anchor_score = float(anchor_score)
        require(np.isfinite(self.anchor_score), "Invalid anchor threshold")
        self.frames, self.events = [], []
        # Retain the objects, not only their Python ids: addresses cannot be recycled
        # while a controller lives. Handles follow the object across numeric-id relabels.
        self._objects = {}
        self._pre_gate_lost = {}
        self._original_update = fc.t.update
        fc.t.update = self._capture_update
        fc.decision_hook = self.decide

    def handle(self, track):
        key = id(track)
        if key not in self._objects:
            self._objects[key] = (track, len(self._objects) + 1)
        stored, handle = self._objects[key]
        require(stored is track, "Track object identity was recycled")
        return handle

    def _capture_update(self, *args, **kwargs):
        outs = self._original_update(*args, **kwargs)
        frame = int(self.fc.frame)
        require(frame == len(self.frames) + 1, "Capture must observe every frame exactly once from frame 1")
        anchors, real_count = [], 0
        for index, track in enumerate(outs):
            handle = self.handle(track)
            if int(track.frame_id) != frame:
                continue
            # At this exact point only BYTETracker has run. _det is therefore a real
            # detector update, even for tracks that were verifier-born on an older frame.
            require(hasattr(track, "_det"), "wrap_stracks must preserve the raw detector box")
            box = np.asarray(track._det, float).copy()
            require(box.shape == (4,) and np.isfinite(box).all(), "Invalid detector observation")
            score = float(track.score)
            require(np.isfinite(score), "Invalid detector score")
            real_count += 1
            if score >= self.anchor_score:
                anchors.append({"handle": handle, "track_id": int(track.track_id),
                                "box": box.tolist(), "score": score, "output_index": index})
        # Resolve fill queries against the lost objects before any gate can mutate
        # them. Keeping this separately also avoids reading pseudo-updated _det later.
        lost = {}
        for track in self.fc.t.lost_stracks:
            tid = int(track.track_id)
            require(tid not in lost, "Duplicate lost-track id before gates")
            lost[tid] = self.handle(track)
        self._pre_gate_lost = lost
        self.frames.append({"sequence": self.name, "frame": frame,
                            "image_relpath": f"img1/{frame:06d}.jpg", "anchor_score": self.anchor_score,
                            "real_output_count_before_score_filter": real_count, "anchors": anchors})
        return outs  # preserve object identities, original order, score dtype and original list

    def decide(self, kind, event):
        require(kind in ("fill", "birth"), "Only the frozen A_orig fill/birth queries are supported")
        frame = int(event["frame"])
        require(self.frames and self.frames[-1]["frame"] == frame, "Query has no pre-gate snapshot")
        event["sequence"], event["kind"] = self.name, kind
        event["signature"] = signature(event)
        tid = event["track_id"]
        if kind == "fill":
            require(tid in self._pre_gate_lost, "Fill query did not originate from a pre-gate lost object")
            handle = self._pre_gate_lost[tid]
        else:
            require(tid is None, "Birth query unexpectedly has a tracked identity")
            handle = None  # no causal identity is inferred for an as-yet-unaccepted birth
        event["query_handle"] = handle
        event["exclude_handles"] = [] if handle is None else [handle]
        event["exclude_track_ids"] = [] if tid is None else [int(tid)]
        event["anchor_handles_after_self_exclusion"] = [
            anchor["handle"] for anchor in self.frames[-1]["anchors"]
            if anchor["handle"] not in event["exclude_handles"]
            and anchor["track_id"] not in event["exclude_track_ids"]]
        self.events.append(event)  # tracker appends its eventual outcome to this same object
        return bool(event["native_accept"])

    def observe(self, frame, rows):
        require(self.frames[-1]["frame"] == frame, "Unexpected observation frame")
        for event in reversed(self.events):
            if event["frame"] != frame:
                break
            if event["outcome"] == "written":
                event["output_ids_after_relabel"] = [
                    int(tid) for tid, box, _ in rows
                    if np.array_equal(np.asarray(box), np.asarray(event["final"]))]

    def finish(self):
        require(all(e["outcome"] != "pending" for e in self.events), "Unfinished query event")
        require(all(e.get("decision_accept") == e["native_accept"] and not e.get("intervened")
                    for e in self.events), "Probe changed an A_orig gate")
        write_jsonl(self.destination / "anchors" / f"{self.name}.jsonl", self.frames)
        write_jsonl(self.destination / "queries" / f"{self.name}.jsonl", self.events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cell", default="v1_dev_last")
    parser.add_argument("--arm", default="A_orig")
    parser.add_argument("--reference-tracks", type=Path, required=True)
    parser.add_argument("--reference-metrics", type=Path,
                        help="Prior canonical A_orig metrics; defaults to reference-tracks/../../metrics/A_orig.json")
    parser.add_argument("--expected-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--anchor-score", type=float, default=.6)
    parser.add_argument("--limit", type=int, default=0, help="Preflight sequence limit; not a formal capture")
    parser.add_argument("--expected-sequences", type=int, default=36)
    parser.add_argument("--expected-fill-count", type=int, default=641)
    parser.add_argument("--expected-birth-count", type=int, default=115)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fctrack.run import run_arm
    from homa.evaluate import evaluate

    root, output = args.root.resolve(), args.output.resolve()
    require(not output.exists(), "Output exists; choose a new immutable execution directory")
    cfg = json.loads(args.config.read_text())
    cell, arm = cfg["cells"][args.cell], cfg["arms"][args.arm]
    require(arm.get("diagnostic", {}).get("mode", "noop") == "noop", "Only the original gate policy is allowed")
    require(not arm.get("verifier_oracle") and arm.get("birth", {}).get("rule") == "verifier",
            "GT-assisted tracker configurations are forbidden")
    require(not arm.get("outlier") and not arm.get("claim"), "Only the frozen A_orig paths are supported")
    controllers = {}

    def factory(name, fc, seq_dir):
        controller = ProbeController(name, fc, seq_dir, output, args.anchor_score)
        controllers[name] = controller
        return controller

    output.mkdir(parents=True, exist_ok=False)
    names, stats, track_hashes = run_arm(root, cfg, cell, arm, output / "tracks", args.device,
                                       args.limit, controller_factory=factory)
    inventory = {str(args.config.resolve()): digest(args.config)}
    totals, per_sequence = Counter(), {}
    for name in names:
        reference = args.reference_tracks / f"{name}.txt"
        expected_file = args.expected_events / f"{name}.jsonl"
        actual_file = output / "queries" / f"{name}.jsonl"
        inventory[str(reference.resolve())] = digest(reference)
        inventory[str(expected_file.resolve())] = digest(expected_file)
        require(reference.read_bytes() == (output / "tracks" / f"{name}.txt").read_bytes(),
                f"Probe changed A_orig tracks: {name}")
        expected, actual = read_queries(expected_file), read_queries(actual_file)
        require(expected == actual, f"Query signatures, order, p, boxes or outcomes changed: {name}")
        controller = controllers[name]
        require(len({(e["kind"], e["frame"], e["candidate_index"], e["signature"]) for e in actual}) == len(actual),
                "Duplicate query address")
        counts = Counter(e["kind"] for e in actual)
        totals.update(counts)
        totals["frames"] += len(controller.frames)
        totals["anchors"] += sum(len(f["anchors"]) for f in controller.frames)
        totals["empty_anchor_frames"] += sum(not f["anchors"] for f in controller.frames)
        per_sequence[name] = {"query_counts": dict(counts), "frames": len(controller.frames),
                              "anchors": sum(len(f["anchors"]) for f in controller.frames),
                              "stable_handle_count": len(controller._objects),
                              "tracks_byte_identical": True, "queries_exactly_identical": True}
        seq = root / cell["dataset_root"] / cell["split"] / name
        det_file = root / cell["detections"] / name / "det.txt"
        for source in (seq / "seqinfo.ini", det_file):
            inventory[str(source.resolve())] = digest(source)
    if not args.limit:
        require(len(names) == args.expected_sequences, "Unexpected formal sequence count")
        require(set(names) == {p.stem for p in args.expected_events.glob("*.jsonl")}, "Expected event sequence set differs")
        require(set(names) == {p.stem for p in args.reference_tracks.glob("*.txt")}, "Reference track sequence set differs")
        require(totals["fill"] == args.expected_fill_count and totals["birth"] == args.expected_birth_count,
                "Frozen fill/birth query counts changed")
    # Evaluation is deliberately after all observations, decisions and output
    # checks have completed. No labels are supplied to ProbeController/run_arm.
    reference_metrics = (args.reference_metrics if args.reference_metrics is not None else
                         args.reference_tracks.parent.parent / "metrics/A_orig.json")
    inventory[str(reference_metrics.resolve())] = digest(reference_metrics)
    prior = json.loads(reference_metrics.read_text())
    canonical_path = output / "canonical_metrics.json"
    canonical = evaluate(root / cell["dataset_root"], cell["split"], output / "tracks", canonical_path, names)
    require(canonical["per_sequence"] == {name: prior["per_sequence"][name] for name in names},
            "Canonical per-sequence metrics changed")
    if not args.limit:
        require(canonical["overall"] == prior["overall"], "Canonical overall metrics changed")
    for name in names:
        gt_file = root / cell["dataset_root"] / cell["split"] / name / "gt/gt.txt"
        inventory[str(gt_file.resolve())] = digest(gt_file)
    verifier = cell["verifiers"][arm["verifier"]]
    weights = root / verifier["weights"]
    inventory[str(weights.resolve())] = digest(weights)
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {str(p.relative_to(source_root)): digest(p) for p in
                     (Path(__file__), source_root / "fctrack/run.py", source_root / "fctrack/tracker.py",
                      source_root / "fctrack/verifier.py", source_root / "yolox/tracker/byte_tracker.py",
                      source_root / "homa/evaluate.py", source_root / "tools/evaluate_uavswarm_mot.py")}
    metadata = {
        "status": "completed", "preflight_only": bool(args.limit), "cell": args.cell, "arm": args.arm,
        "dataset_root": cell["dataset_root"], "split": cell["split"],
        "dataset_sequence_root": str((root / cell["dataset_root"] / cell["split"]).resolve()),
        "frame_origin": "Exp089 dev_view reindexed frame numbers; frame 1 is each view's first frame",
        "gt_files_read_by_capture": False, "gt_read_by_posthoc_evaluator": True,
        "reference_event_gt_labels_used": False,
        "image_inputs": "same lazy image calls and verifier batches as A_orig; external run input inventory records image hashes",
        "anchor_score": args.anchor_score, "anchor_phase": "immediately after BYTETracker.update, before FCTrack recover/birth/relabel",
        "anchor_box": "raw detector _det in original-image tlwh coordinates; deep-copied before pseudo updates",
        "anchor_continuity": "per-sequence stable STrack object handle; strong references prevent Python id reuse; numeric track_id can change or be reused by relabel",
        "anchor_limit": "A real detector observation is not proof of a correct identity or true target; threshold .6 is an observation filter only",
        "self_exclusion": "At both current and historical frames exclude all query exclude_handles and exclude_track_ids; pair remaining observations only by handle",
        "birth_identity": "null query_handle and empty exclusions: no pre-existing identity is inferred for a birth query",
        "history_boundary": "all frames including empty-anchor frames are stored; history before frame 1 is missing, never repeated first-frame evidence",
        "counts": dict(totals), "sequence_count": len(names), "tracks_byte_identical": [len(names), len(names)],
        "all_queries_exactly_identical": True, "track_sha256": track_hashes, "tracker_stats": stats,
        "canonical_metrics_path": "canonical_metrics.json", "canonical_metrics_sha256": digest(canonical_path),
        "canonical_per_sequence_matches_reference": True,
        "canonical_overall_matches_reference": True if not args.limit else None,
        "source_sha256": source_hashes, "input_sha256": inventory, "per_sequence": per_sequence,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "completed", "sequence_count": len(names), "counts": dict(totals),
                      "tracks_byte_identical": [len(names), len(names)], "all_queries_exactly_identical": True}), flush=True)


if __name__ == "__main__":
    main()
