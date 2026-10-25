import unittest
from types import SimpleNamespace

import numpy as np

from fctrack.reid import crop, fuse
from yolox.tracker.byte_tracker import STrack, BYTETracker
from yolox.tracker.kalman_filter import KalmanFilter


class ReIDTests(unittest.TestCase):
    def test_recovery_cannot_refresh_gallery(self):
        t = STrack([0, 0, 10, 10], .9)
        t.update_reid([1., 0.])
        t.activate(KalmanFilter(), 1)
        t.re_activate(STrack([1, 0, 10, 10], .7), 2)
        np.testing.assert_array_equal(t.reid_feature, [1, 0])
        d = STrack([1, 0, 10, 10], .9)
        d.update_reid([0., 1.])
        t.update(d, 3)
        np.testing.assert_allclose(t.reid_feature, np.array([.9, .1]) / np.linalg.norm([.9, .1]), rtol=1e-6)

    def test_gates_and_fallback(self):
        t = STrack([0, 0, 10, 10], .9)
        t.update_reid([1, 0])
        dets = [STrack([0, 0, 10, 10], .9), STrack([100, 0, 10, 10], .9), STrack([0, 0, 10, 10], .9)]
        dets[0].update_reid([1, 0]); dets[1].update_reid([1, 0]); dets[2].update_reid([-1, 0])
        spec = dict(proximity=.5, appearance=.25)
        np.testing.assert_allclose(fuse([t], dets, np.full((1, 3), .8), spec), [[0, .8, .8]])
        del t.reid_feature
        np.testing.assert_allclose(fuse([t], dets, np.full((1, 3), .8), spec), .8)

    def test_invalid_crop_and_feature(self):
        self.assertIsNone(crop(np.zeros((10, 10, 3), np.uint8), [20, 20, 2, 2]))
        t = STrack([0, 0, 10, 10], .9)
        t.update_reid([0, 0]); t.update_reid([float('nan'), 1])
        self.assertFalse(hasattr(t, 'reid_feature'))

    def test_detection_count_alignment(self):
        t = BYTETracker(SimpleNamespace(track_thresh=.6, track_buffer=30, match_thresh=.9, mot20=False))
        t.reid_spec = dict(proximity=.5, appearance=.25)
        t.reid_current = np.empty((0, 2))
        with self.assertRaisesRegex(ValueError, 'alignment'):
            t.update(np.array([[0, 0, 10, 10, .9]]), (100, 100), (100, 100))


if __name__ == '__main__':
    unittest.main()
