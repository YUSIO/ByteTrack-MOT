import unittest

import numpy as np

from yolox.tracker.anchor_displacement import associate_with_anchor_displacement, weighted_median
from yolox.tracker.basetrack import TrackState


class DummyTrack:
    def __init__(self, track_id, center, frame_id=0, tracklet_len=3):
        self.track_id = track_id
        self._tlwh = np.asarray([center[0] - 1.0, center[1] - 1.0, 2.0, 2.0], dtype=float)
        self.state = TrackState.Tracked
        self.frame_id = frame_id
        self.tracklet_len = tracklet_len

    @property
    def tlwh(self):
        return self._tlwh.copy()


class DummyDetection:
    def __init__(self, center, score=0.9):
        self._tlwh = np.asarray([center[0] - 1.0, center[1] - 1.0, 2.0, 2.0], dtype=float)
        self.score = score

    @property
    def tlwh(self):
        return self._tlwh.copy()


class AnchorDisplacementTest(unittest.TestCase):
    def test_weighted_median_is_deterministic(self):
        values = np.asarray([[0.0, 4.0], [10.0, 1.0], [11.0, 8.0]])
        result = weighted_median(values, np.asarray([1.0, 3.0, 1.0]))
        np.testing.assert_allclose(result, [10.0, 1.0])

    def _ambiguous_case(self):
        tracks = [
            DummyTrack(0, (10.0, 0.0)),
            DummyTrack(1, (10.0, 10.0)),
            DummyTrack(2, (0.0, 0.0)),
            DummyTrack(3, (0.0, 10.0)),
            DummyTrack(4, (10.0, 20.0)),
        ]
        detections = [
            DummyDetection((15.0, 2.0)),
            DummyDetection((15.0, 12.0)),
            DummyDetection((5.0, 2.0)),
            DummyDetection((5.0, 12.0)),
            DummyDetection((15.0, 22.0)),
        ]
        inf = np.inf
        base_cost = np.asarray(
            [
                [0.30, 0.10, inf, inf, inf],
                [0.10, 0.30, inf, inf, inf],
                [inf, inf, 0.10, inf, inf],
                [inf, inf, inf, 0.10, inf],
                [inf, inf, inf, inf, 0.10],
            ],
            dtype=float,
        )
        base_matches = np.asarray([[0, 1], [1, 0], [2, 2], [3, 3], [4, 4]], dtype=int)
        prior_centers = {track.track_id: track.tlwh[:2] + track.tlwh[2:4] / 2.0 for track in tracks}
        return tracks, detections, base_cost, base_matches, prior_centers

    def test_anchor_displacement_repairs_swapped_provisional_match(self):
        tracks, detections, base_cost, base_matches, prior_centers = self._ambiguous_case()
        matches, unmatched_tracks, unmatched_detections, traces = associate_with_anchor_displacement(
            tracks,
            detections,
            base_cost,
            base_matches,
            prior_centers,
            frame_id=1,
            match_threshold=0.9,
            anchor_min_count=3,
            anchor_max_count=3,
            anchor_min_age=3,
            anchor_lambda=0.25,
        )
        self.assertEqual(set(map(tuple, matches.tolist())), {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)})
        self.assertEqual(unmatched_tracks, ())
        self.assertEqual(unmatched_detections, ())
        self.assertTrue(any(item["targets"] for item in traces))

    def test_insufficient_anchors_preserve_baseline(self):
        tracks, detections, base_cost, base_matches, prior_centers = self._ambiguous_case()
        matches, _, _, _ = associate_with_anchor_displacement(
            tracks[:3],
            detections[:3],
            base_cost[:3, :3],
            base_matches[:2],
            {key: value for key, value in prior_centers.items() if key < 3},
            frame_id=1,
            match_threshold=0.9,
            anchor_min_count=3,
            anchor_min_age=3,
        )
        np.testing.assert_array_equal(matches, base_matches[:2])


if __name__ == "__main__":
    unittest.main()
