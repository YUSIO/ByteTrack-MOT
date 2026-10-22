"""Association boundaries and lifecycle behavior for the fixed-cache adapter."""
import unittest
from types import SimpleNamespace
import numpy as np
from yolox.tracker.joint_association import iterative_assignment
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.basetrack import BaseTrack


class JointAssociationTests(unittest.TestCase):
    def test_ties_empty_and_strict_boundary(self):
        for shape in [(0, 3), (3, 0), (0, 0)]:
            m, t, d = iterative_assignment(np.zeros(shape), .9)
            self.assertEqual(m.shape, (0, 2))
            self.assertEqual((len(t), len(d)), shape)
        m, _, _ = iterative_assignment(np.full((2, 2), .2), .9)
        np.testing.assert_array_equal(m, [[0, 0], [1, 1]])
        self.assertEqual(len(iterative_assignment([[.9]], .9)[0]), 0)

    def test_threshold_reduces_each_round(self):
        # First round picks (0,0); the remaining .87 is below initial .9,
        # but above second-round .85 and must remain unmatched.
        m, t, d = iterative_assignment([[.1, .8], [.2, .87]], .9)
        np.testing.assert_array_equal(m, [[0, 0]])
        self.assertEqual((t, d), ((1,), (1,)))

    def tracker(self, mode=None):
        BaseTrack._count = 0
        args = dict(track_thresh=.6, track_buffer=30, match_thresh=.9, mot20=False)
        if mode:
            args['joint_association'] = dict(mode=mode, low_penalty=.2, reduce_step=.05)
        return BYTETracker(SimpleNamespace(**args))

    def step(self, t, score=None):
        det = np.array([[10, 10, 20, 20, score]], dtype=float) if score else np.empty((0, 5))
        return t.update(det, (100, 100), (100, 100))

    def test_lost_can_reactivate_with_low_without_new_id(self):
        for mode in ['tracktrack', 'hungarian']:
            t = self.tracker(mode)
            tid = self.step(t, .9)[0].track_id
            self.assertFalse(self.step(t))
            out = self.step(t, .4)
            self.assertEqual([x.track_id for x in out], [tid])
            self.assertEqual(t.joint_stats['lost_low'], 1)
            self.assertFalse(t.lost_stracks)
        t = self.tracker()
        self.step(t, .9)
        self.step(t)
        self.assertFalse(self.step(t, .4))

    def test_low_cannot_birth_or_confirm(self):
        for mode in ['tracktrack', 'hungarian']:
            t = self.tracker(mode)
            self.assertFalse(self.step(t, .4))
            self.assertFalse(t.tracked_stracks)
            self.assertFalse(self.step(t, .9))  # new at frame 2 -> unconfirmed
            self.assertFalse(self.step(t, .4))  # only high may confirm
            self.assertFalse(t.tracked_stracks)

    def test_original_score_preserved(self):
        for mode in ['tracktrack', 'hungarian']:
            t = self.tracker(mode)
            self.step(t, .9)
            out = self.step(t, .4)
            self.assertEqual(out[0].score, .4)


if __name__ == '__main__':
    unittest.main()
