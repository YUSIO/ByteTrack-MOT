"""Causal read boundary, anchor semantics, and endpoint-only wrapper checks."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
import torch

from fctrack.visual_correspondence import VisualCorrespondence


class FakePredictor:
    step = 8
    interp_shape = (384, 512)

    def __init__(self, offset=1.):
        self.offset = offset
        self.chunks, self.initial_queries = [], None
        self.total = 0

    def __call__(self, video_chunk, is_first_step=False, queries=None, add_support_grid=False):
        assert add_support_grid
        values = video_chunk[0, :, 0, 0, 0].cpu().numpy().tolist()
        if is_first_step:
            self.initial_queries = queries.clone()
            self.total = 0
            return None, None
        self.chunks.append(values)
        t = video_chunk.shape[1]
        self.total = t if self.total == 0 else self.total + min(self.step, t - self.step)
        xy = self.initial_queries[:, :, 1:]
        coords = xy[:, None].repeat(1, self.total, 1, 1) + self.offset
        # Deliberately false: the provider must still return the coordinate.
        visible = torch.zeros(coords.shape[:3], dtype=torch.bool, device=coords.device)
        return coords, visible


class VisualCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'img1').mkdir()
        (self.root / 'seqinfo.ini').write_text('[Sequence]\nimDir=img1\nimExt=.png\n')
        for f in range(1, 20):
            cv2.imwrite(str(self.root / 'img1' / f'{f:06d}.png'),
                        np.full((30, 40, 3), f, dtype=np.uint8))
        # Target anchor center(20,15), one valid support, one low, future support.
        self.det = np.array([[2, -1, 18, 13, 4, 4, .9],
                             [2, -1, 26, 13, 4, 4, .9],
                             [2, -1, 10, 13, 4, 4, .4],
                             [18, -1, 20, 13, 4, 4, .99]], dtype=float)
        self.track = SimpleNamespace(track_id=7, _visual_anchor_frame=2,
                                     _visual_anchor_box=np.array([18., 13., 4., 4.]))

    def tearDown(self):
        self.tmp.cleanup()

    def provider(self, model=None):
        return VisualCorrespondence({'max_history': 30, 'crop_wh': [512, 384]},
                                    self.root, self.det, 'cpu', predictor=model or FakePredictor())

    def test_short_prefix_and_false_visibility_is_not_gate(self):
        p = self.provider()
        result = p(7, self.track)
        np.testing.assert_array_equal(result, [21., 16.])
        self.assertEqual(p.model.chunks, [[2., 3., 4., 5., 6., 7.]])
        self.assertEqual(p.records[-1]['input_frames'], list(range(2, 8)))
        self.assertEqual(p.records[-1]['support_detection_count'], 1)
        self.assertFalse(p.records[-1]['visibility_confidence_flag'])
        self.assertTrue(p.records[-1]['no_future_frames'])
        np.testing.assert_array_equal(p.model.initial_queries[0, 0].numpy(), [0., 20., 15.])

    def test_long_prefix_covers_exact_endpoint_without_future(self):
        p = self.provider()
        p(18, self.track)
        self.assertEqual(p.model.chunks, [list(map(float, range(2, 18))),
                                         list(map(float, range(10, 19)))])
        self.assertEqual(p.records[-1]['latest_input_frame'], 18)
        self.assertNotIn(19, p.cache)

    def test_real_anchor_box_is_used_instead_of_current_state(self):
        self.track.tlwh = np.array([500., 500., 4., 4.])
        p = self.provider()
        np.testing.assert_array_equal(p(7, self.track), [21., 16.])
        np.testing.assert_array_equal(self.track._visual_anchor_box, [18., 13., 4., 4.])

    def test_missing_old_anchor_and_invalid_image_request(self):
        p = self.provider()
        self.assertIsNone(p(7, SimpleNamespace(track_id=4)))
        self.assertEqual(p.records[-1]['fallback'], 'missing_anchor')
        self.assertIsNone(p(33, self.track))
        self.assertEqual(p.records[-1]['fallback'], 'anchor_age')
        self.assertFalse(p.cache)
        with self.assertRaises(ValueError):
            p._read_frame(8, 7)

    def test_escaped_crop_and_nonfinite_endpoint_fall_back(self):
        for offset, reason in [(100., 'escaped_crop'), (float('nan'), 'nonfinite_endpoint')]:
            p = self.provider(FakePredictor(offset))
            self.assertIsNone(p(7, self.track))
            self.assertEqual(p.records[-1]['fallback'], reason)


if __name__ == '__main__':
    unittest.main()
