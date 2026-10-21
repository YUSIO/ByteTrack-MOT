"""Decision-only diagnostics must preserve scores, competition and state semantics."""
import copy
import unittest
from types import SimpleNamespace

import numpy as np

from fctrack.tracker import FCTrack


class Track:
    count = 100

    def __init__(self, box, score, tid=None):
        self.tlwh = np.asarray(box, float).copy()
        self.score = score
        self.track_id = tid
        self.frame_id = 1
        self.is_activated = tid is not None
        self.covariance = np.eye(8)

    def re_activate(self, new, frame, new_id=False):
        self.tlwh = new.tlwh.copy()
        self.score = new.score
        self.frame_id = frame
        self.is_activated = True

    def activate(self, kalman, frame):
        Track.count += 1
        self.track_id = Track.count
        self.frame_id = frame


RECOVER = {"estimator": "hold", "lookback": 5, "max_since_real": 30,
           "min_observations": 1, "skip_if_detected": False, "iou_free": .3,
           "p_min": .97, "use_regressed_box": True, "score": .7}
BIRTH = {"lo": .6, "hi": 1.01, "cover_iou": .3, "rule": "verifier",
         "p_min": .97, "use_regressed_box": True}


def make_fc(probabilities, regressed, fill=True):
    Track.count = 100
    tracker = SimpleNamespace(tracked_stracks=[], lost_stracks=[], removed_stracks=[], kalman_filter=None)
    tracker.update = lambda *args: []
    cfg = {"recover": copy.deepcopy(RECOVER)} if fill else {"birth": copy.deepcopy(BIRTH)}
    verify = lambda img, boxes: (np.asarray(probabilities, float).copy(), np.asarray(regressed, float).copy())
    fc = FCTrack(tracker, Track, cfg, verify, log=True)
    fc.frame = 1
    if fill:
        for i in range(len(probabilities)):
            t = Track([30 * i, 20, 10, 10], .8, i + 1)
            tracker.lost_stracks.append(t)
            fc.meta[t.track_id] = {"real": 1, "obs": 2, "box": t.tlwh.copy()}
    return fc


def fingerprint(fc, rows):
    return {
        "rows": [(int(tid), tuple(box), float(score)) for tid, box, score in rows],
        "tracked": [(t.track_id, t.frame_id, tuple(t.tlwh), float(t.score)) for t in fc.t.tracked_stracks],
        "lost": [t.track_id for t in fc.t.lost_stracks],
        "meta": {k: {a: tuple(b) if isinstance(b, np.ndarray) else b for a, b in v.items()} for k, v in fc.meta.items()},
        "stats": dict(fc.stats), "attempts": fc.attempts,
    }


class DecisionHookTests(unittest.TestCase):
    def test_no_hook_and_identity_hook_have_same_fill_state_and_output(self):
        args = ([.98, .99, .2], [[0, 0, 10, 10], [0, 0, 10, 10], [100, 0, 10, 10]])
        native = make_fc(*args)
        expected = fingerprint(native, native.step(np.empty((0, 5)), (200, 200), lambda: None))
        observed = []
        hooked = make_fc(*args)
        hooked.decision_hook = lambda kind, e: observed.append((kind, e)) or e["native_accept"]
        actual = fingerprint(hooked, hooked.step(np.empty((0, 5)), (200, 200), lambda: None))
        self.assertEqual(expected, actual)
        self.assertEqual([e["outcome"] for _, e in observed], ["overlap", "written", "below_p"])
        self.assertEqual([e["track_id"] for _, e in observed], [1, 2, 3])
        self.assertEqual([e["candidate_index"] for _, e in observed], [0, 1, 2])
        self.assertEqual(observed[1][1]["track_id_after"], 2)
        hooked.t.tracked_stracks[0].track_id = 88
        self.assertEqual(observed[1][1]["track_id"], 2)

    def test_fill_veto_does_not_replace_probabilities_or_ranking(self):
        fc = make_fc([.98, .99, .999], [[0, 0, 10, 10], [0, 0, 10, 10], [100, 0, 10, 10]])
        events = []
        fc.decision_hook = lambda kind, e: events.append(e) or (e["track_id"] != 3 and e["native_accept"])
        rows = fc.step(np.empty((0, 5)), (200, 200), lambda: None)
        self.assertEqual([r[0] for r in rows], [2])
        self.assertEqual([e["p"] for e in events], [.98, .99, .999])
        self.assertEqual([e["outcome"] for e in events], ["overlap", "written", "below_p"])
        self.assertTrue(events[2]["intervened"])

    def test_fill_admit_keeps_geometry_competition_and_written_score(self):
        fc = make_fc([.99, .2, .1, .95], [[0, 0, 10, 10], [100, 0, 10, 10],
                                       [150, 0, 0, 10], [0, 0, 10, 10]])
        events = []
        fc.decision_hook = lambda kind, e: events.append(e) or True
        rows = fc.step(np.empty((0, 5)), (200, 200), lambda: None)
        self.assertEqual([r[0] for r in rows], [1, 2])
        self.assertEqual([r[2] for r in rows], [.7, .7])
        self.assertEqual([e["outcome"] for e in events], ["written", "written", "degenerate", "overlap"])
        self.assertEqual(fc.meta[2]["real"], 1)
        self.assertEqual(events[1]["gap"], 1)
        self.assertEqual(events[1]["since_real"], 1)
        self.assertEqual(events[1]["obs"], 2)

    def test_identity_birth_preserves_legacy_filter_then_sort(self):
        det = np.array([[0, 0, 10, 10, .2], [0, 0, 10, 10, .8],
                        [100, 0, 110, 10, .9], [150, 0, 160, 10, .8]], np.float32)
        args = ([.98, .2, .99], [[0, 0, 10, 10], [100, 0, 10, 10], [0, 0, 10, 10]])
        native = make_fc(*args, fill=False)
        expected = fingerprint(native, native.step(det, (200, 200), lambda: None))
        observed = []
        hooked = make_fc(*args, fill=False)
        hooked.decision_hook = lambda kind, e: observed.append(e) or e["native_accept"]
        actual = fingerprint(hooked, hooked.step(det, (200, 200), lambda: None))
        self.assertEqual(expected, actual)
        self.assertEqual([e["candidate_index"] for e in observed], [0, 1, 2])
        self.assertEqual([e["detection_index"] for e in observed], [1, 2, 3])
        self.assertEqual([e["outcome"] for e in observed], ["written", "below_p", "overlap"])
        self.assertTrue(all(e["track_id"] is None for e in observed))

    def test_birth_admit_preserves_detection_score_and_rejects_bad_geometry(self):
        det = np.array([[0, 0, 10, 10, .65], [100, 0, 110, 10, .7],
                        [150, 0, 160, 10, .8], [0, 0, 10, 10, .9]], np.float32)
        fc = make_fc([.98, .2, .1, .99], [[0, 0, 10, 10], [100, 0, 10, 10],
                                       [150, 0, 0, 10], [0, 0, 10, 10]], fill=False)
        events = []
        fc.decision_hook = lambda kind, e: events.append(e) or True
        rows = fc.step(det, (200, 200), lambda: None)
        self.assertEqual([float(r[2]) for r in rows], [float(det[3, 4]), float(det[1, 4])])
        self.assertEqual([e["outcome"] for e in events], ["overlap", "written", "degenerate", "written"])
        self.assertEqual([e["track_id_after"] for e in events if e["outcome"] == "written"], [102, 101])
        self.assertEqual(events[1]["p"], .2)
        self.assertEqual(fc.meta[102]["prov"], 0)

    def test_unsupported_hook_configuration_fails_before_step_mutates_state(self):
        for field, value in [("estimator", "hold+kalman"), ("iterations", 2), ("write", False)]:
            fc = make_fc([.99], [[0, 0, 10, 10]])
            fc.rec[field] = value
            fc.decision_hook = lambda kind, e: e["native_accept"]
            with self.assertRaises(ValueError):
                fc.step(np.empty((0, 5)), (200, 200), lambda: None)
            self.assertEqual(fc.frame, 1)
        fc = make_fc([.99], [[0, 0, 10, 10]], fill=False)
        fc.bth["rule"] = "oracle"
        fc.decision_hook = lambda kind, e: e["native_accept"]
        with self.assertRaises(ValueError):
            fc.step(np.empty((0, 5)), (200, 200), lambda: None)

    def test_hook_must_explicitly_return_boolean(self):
        fc = make_fc([.99], [[0, 0, 10, 10]])
        fc.decision_hook = lambda kind, e: None
        with self.assertRaises(TypeError):
            fc.step(np.empty((0, 5)), (200, 200), lambda: None)


if __name__ == "__main__":
    unittest.main()
