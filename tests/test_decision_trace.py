import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from fctrack.decision_trace import DecisionTrace, plain
from fctrack.tracker import FCTrack, wrap_stracks
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.byte_tracker import BYTETracker, STrack


class DecisionTraceTests(unittest.TestCase):
    def replay(self, path=None):
        wrap_stracks(STrack)
        BaseTrack._count = 0
        tracker = BYTETracker(SimpleNamespace(track_thresh=.6, track_buffer=3, match_thresh=.9, mot20=False))
        cfg = dict(recover=dict(estimator='kalman', lookback=2, max_since_real=3,
                               min_observations=1, iou_free=.3, skip_if_detected=False,
                               p_min=.97, use_regressed_box=True, score=.7),
                   birth=dict(lo=.6, hi=1.01, cover_iou=.3, rule='verifier', p_min=.97, use_regressed_box=True))
        calls = []
        def verify(img, boxes):
            calls.append((fc.frame, boxes.copy().tolist()))
            return np.full(len(boxes), 1. if fc.frame in (3, 6) else .8), boxes.copy()
        fc = FCTrack(tracker, STrack, cfg, verify, log=True)
        if path is not None:
            fc.trace = tracker.decision_trace = DecisionTrace(path)
        frames = [
            [[10,10,20,20,.9], [60,10,70,20,.9], [80,80,90,90,.04]],
            [[11,10,21,20,.2]],
            [[12,10,22,20,.3], [60,10,70,20,.3]],
            [], [], [[80,60,90,70,.65]], [], [], [], [],
        ]
        output = []
        states = []
        for f, values in enumerate(frames, 1):
            data = np.asarray(values, np.float32).reshape(-1,5)
            original = data.copy()
            for tid, b, score in fc.step(data, (100,100), lambda: None):
                output.append((f, tid, *(round(v, 1) for v in b), round(score, 2)))
            np.testing.assert_array_equal(original, data)
            states.append({pool: [(t.track_id, t.state, t.frame_id, t.mean.tolist())
                                  for t in getattr(tracker, pool+'_stracks')]
                           for pool in ('tracked','lost','removed')})
        if fc.trace is not None:
            fc.trace.close()
        return plain(output), states, calls, dict(fc.stats)

    def test_instrumentation_preserves_outputs_states_and_verifier_calls(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'trace.jsonl.gz'
            baseline = self.replay()
            observed = self.replay(path)
            self.assertEqual(baseline, observed)
            with gzip.open(path, 'rt') as stream:
                records = [json.loads(line) for line in stream]
            self.assertEqual([r['frame'] for r in records], list(range(1,11)))
            stages = {e['stage'] for r in records for e in r['events']}
            self.assertTrue({'association_low','birth_verification','recovery_verification','byte_remove'} <= stages)
            low_gate = next(e for e in records[2]['events'] if e['stage']=='low_eligibility')
            self.assertTrue(any(not r['eligible'] and r['track']['state_name']=='Lost' for r in low_gate['tracks']))
            low = next(e for e in records[2]['events'] if e['stage']=='association_low')
            self.assertEqual(low['detection_indices'], [0,1])
            self.assertEqual(low['matches'], [[0,0]])
            high = next(e for e in records[0]['events'] if e['stage']=='detection_bands')
            self.assertEqual(high['high_indices'], [0,1])
            self.assertEqual(high['ignored_indices'], [2])
            rejects = [e for r in records for e in r['events'] if e['stage']=='recovery_eligibility']
            self.assertTrue(any('gap_outside_lookback' in e['reasons'] for e in rejects))
            self.assertTrue(any('since_real_exceeded' in e['reasons'] for e in rejects))

    def test_uid_survives_relabel_and_events_snapshot_arrays(self):
        with TemporaryDirectory() as tmp:
            trace = DecisionTrace(Path(tmp)/'trace.jsonl.gz')
            BaseTrack._count=0
            t=STrack([1,2,3,4], .9)
            t.activate(BYTETracker(SimpleNamespace(track_thresh=.6,track_buffer=30,match_thresh=.9,mot20=False)).kalman_filter, 1)
            before_attributes=set(t.__dict__)
            uid=trace.uid(t)
            old=trace.track(t)
            t.track_id=99
            t.mean[0]+=10
            self.assertEqual(trace.uid(t),uid)
            self.assertEqual(old['track_id'],1)
            self.assertNotEqual(old['mean'],t.mean.tolist())
            self.assertEqual(set(t.__dict__),before_attributes)
            other=STrack([1,2,3,4], .9)
            self.assertNotEqual(trace.uid(other),uid)
            trace.close()


if __name__=='__main__':
    unittest.main()
