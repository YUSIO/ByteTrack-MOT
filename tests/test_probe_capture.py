"""Capture must freeze detector evidence before pseudo writes and relabels."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from fctrack.probe_capture import ProbeController, read_queries, signature


def track(tid, frame=1, score=.8, box=(10, 20, 8, 6)):
    return SimpleNamespace(track_id=tid, frame_id=frame, score=score, _det=np.array(box, float))


def event(kind="fill", tid=1, frame=1, accept=True):
    return {"frame": frame, "candidate_index": 0, "track_id": tid if kind == "fill" else None,
            "proposal": [10., 20., 8., 6.], "final": [11., 21., 8., 6.],
            "p": .99 if accept else .1, "native_accept": accept, "threshold": .97, "outcome": "pending"}


class CaptureTests(unittest.TestCase):
    def make(self, outs, lost=()):
        holder = {"outs": outs}
        tracker = SimpleNamespace(lost_stracks=list(lost))
        tracker.update = lambda *args, **kwargs: holder["outs"]
        fc = SimpleNamespace(t=tracker, frame=1, decision_hook=None)
        controller = ProbeController("seq", fc, "/does/not/contain/gt", "/unused", .6)
        return fc, controller, holder

    def test_returns_original_list_and_snapshots_raw_box_before_mutation(self):
        observed = track(1)
        outs = [observed]
        fc, cap, _ = self.make(outs)
        result = fc.t.update(None)
        self.assertIs(result, outs)
        self.assertIs(result[0], observed)
        observed._det[:] = -999
        observed.track_id = 9
        self.assertEqual(cap.frames[0]["anchors"][0]["box"], [10., 20., 8., 6.])
        self.assertEqual(cap.frames[0]["anchors"][0]["track_id"], 1)

    def test_score_and_real_frame_filter_and_empty_frame_are_explicit(self):
        fc, cap, holder = self.make([track(1, score=.59), track(2, score=.6), track(3, frame=0)])
        fc.t.update()
        self.assertEqual([a["track_id"] for a in cap.frames[0]["anchors"]], [2])
        self.assertEqual(cap.frames[0]["real_output_count_before_score_filter"], 2)
        holder["outs"] = []
        fc.frame = 2
        fc.t.update()
        self.assertEqual(cap.frames[1]["anchors"], [])
        self.assertEqual(cap.frames[1]["image_relpath"], "img1/000002.jpg")

    def test_handle_follows_object_not_mutable_numeric_id(self):
        first = track(7)
        fc, cap, holder = self.make([first])
        fc.t.update()
        h1 = cap.frames[0]["anchors"][0]["handle"]
        first.track_id, first.frame_id = 3, 2
        second = track(7, frame=2)
        holder["outs"] = [first, second]
        fc.frame = 2
        fc.t.update()
        self.assertEqual(cap.frames[1]["anchors"][0]["handle"], h1)
        self.assertNotEqual(cap.frames[1]["anchors"][1]["handle"], h1)
        self.assertIs(cap._objects[id(first)][0], first)

    def test_fill_uses_frozen_lost_handle_and_excludes_both_handle_and_id(self):
        own = track(7, frame=0)
        alias = track(7)
        neighbor = track(8)
        fc, cap, _ = self.make([alias, neighbor], [own])
        fc.t.update()
        ev = event(tid=7)
        own.track_id = 99  # later mutation cannot alter the frozen pre-gate identity lookup
        self.assertTrue(fc.decision_hook("fill", ev))
        self.assertEqual(ev["query_handle"], cap.handle(own))
        self.assertEqual(ev["exclude_track_ids"], [7])
        self.assertEqual(ev["anchor_handles_after_self_exclusion"], [cap.handle(neighbor)])

    def test_birth_has_no_inferred_identity_and_native_rejection_is_preserved(self):
        observed = track(1)
        fc, cap, _ = self.make([observed])
        fc.t.update()
        ev = event("birth", accept=False)
        self.assertFalse(fc.decision_hook("birth", ev))
        self.assertIsNone(ev["query_handle"])
        self.assertEqual(ev["exclude_handles"], [])
        self.assertEqual(ev["anchor_handles_after_self_exclusion"], [cap.handle(observed)])

    def test_missing_frame_or_missing_fill_identity_fails(self):
        fc, cap, _ = self.make([])
        with self.assertRaisesRegex(RuntimeError, "no pre-gate"):
            fc.decision_hook("fill", event())
        fc.t.update()
        with self.assertRaisesRegex(RuntimeError, "pre-gate lost"):
            fc.decision_hook("fill", event())
        fc.frame = 3
        with self.assertRaisesRegex(RuntimeError, "every frame"):
            fc.t.update()

    def test_observe_and_finish_preserve_signature_and_skip_reference_gt_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            own = track(1, frame=0)
            fc, cap, _ = self.make([], [own])
            cap.destination = Path(directory)
            fc.t.update()
            ev = event()
            expected_signature = signature(ev)
            fc.decision_hook("fill", ev)
            ev.update(outcome="written", decision_accept=True, intervened=False, track_id_after=1)
            cap.observe(1, [(3, np.array(ev["final"]), .7)])
            self.assertEqual(ev["signature"], expected_signature)
            self.assertEqual(ev["output_ids_after_relabel"], [3])
            cap.finish()
            path = cap.destination / "queries/seq.jsonl"
            record = json.loads(path.read_text())
            self.assertNotIn("any_hit", record)
            record["any_hit"] = True
            record["owner_gt_id"] = 9
            reference = Path(directory) / "reference.jsonl"
            reference.write_text(json.dumps(record) + "\n")
            self.assertEqual(read_queries(path), read_queries(reference))
            with self.assertRaises(FileExistsError):
                cap.finish()


if __name__ == "__main__":
    unittest.main()
