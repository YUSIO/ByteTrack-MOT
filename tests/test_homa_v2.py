import unittest
from types import SimpleNamespace
import numpy as np
import torch
from homa.model import PaperPoseBlock, association_loss, window_association_loss
from homa.association import HOMAAssociation
from homa.tracker import MHATracker
from yolox.tracker.basetrack import BaseTrack


class RevisionTests(unittest.TestCase):
    def test_eq4_is_unscaled_and_float32_under_autocast(self):
        torch.manual_seed(42)
        m = PaperPoseBlock(16, heads=4, hidden=12).eval()
        c, h = torch.randn(2, 16), torch.randn(3, 16)
        with torch.no_grad():
            hp = m.hmlp(m.sa(h[None], h[None], h[None], need_weights=False)[0])
            cp = m.cmlp(m.ca(c[None], hp, hp, need_weights=False)[0])
            expected = cp[0] @ hp[0].T
            with torch.autocast('cpu', dtype=torch.bfloat16):
                actual = m(c, h)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected)

    def test_all_frames_query_without_self_frame_leakage(self):
        class Spy:
            def __init__(self): self.pairs = []
            def logits(self, current, history):
                self.pairs.append((current[:, 0].tolist(), history[:, 0].tolist()))
                logits = current @ history.T
                return torch.stack([logits] * 3), logits
        spy = Spy()
        embeddings = torch.tensor([[1., 0.], [1., 1.], [2., 0.], [2., 1.], [3., 0.], [3., 1.]], requires_grad=True)
        ids = torch.tensor([7, 8, 7, 8, 7, 8]); frames = torch.tensor([1, 1, 2, 2, 3, 3])
        loss, n = window_association_loss(spy, embeddings, ids, frames)
        self.assertEqual(n, 12)
        self.assertEqual(len(spy.pairs), 3)
        for q, h in spy.pairs: self.assertTrue(set(q).isdisjoint(h))
        loss.backward(); self.assertTrue(torch.isfinite(embeddings.grad).all())

    def test_pair_weighted_reduction_handles_unequal_presence(self):
        parts = torch.zeros(3, 2, 3, requires_grad=True)
        fused = torch.zeros(2, 3, requires_grad=True)
        loss, n = association_loss(parts, fused, torch.tensor([1, 2]), torch.tensor([1, 2, 1]),
                                   torch.tensor([1, 1, 2]), pair_weighted=True)
        self.assertEqual(n, 3)
        self.assertAlmostEqual(float(loss.detach()), 8 * np.log(2) / 3, places=6)
        loss.backward(); self.assertGreater(float(fused.grad.abs().sum()), 0)

    def make_tracker(self):
        BaseTrack._count = 0
        t = MHATracker(SimpleNamespace(track_thresh=.6, match_thresh=.9))
        t.homa = HOMAAssociation(None, 'mha_iou', window=8)
        return t

    def step(self, t, score=None):
        t.homa.set_current(None, t.frame_id + 1)
        det = np.empty((0, 5), dtype=np.float32) if score is None else np.array([[0, 0, 20, 20, score]], dtype=np.float32)
        return t.update(det, (100, 100), (100, 100))

    def test_all_high_detections_birth_immediately_and_match(self):
        t = self.make_tracker(); a = self.step(t, .65); b = self.step(t, .9)
        self.assertEqual(len(a), 1); self.assertTrue(a[0].is_activated)
        self.assertEqual(a[0].track_id, b[0].track_id)

    def test_low_track_is_excluded_from_high_pool_but_low_recovers(self):
        t = self.make_tracker(); a = self.step(t, .9); b = self.step(t, .4)
        observed = []; original = t.homa.cost
        def capture(tracks, detections):
            observed.extend(x.track_id for x in tracks)
            return original(tracks, detections)
        t.homa.cost = capture
        c = self.step(t, .4)
        self.assertEqual(observed, [])
        self.assertEqual([x.track_id for x in c], [a[0].track_id])

    def test_lost_track_expires_at_temporal_window(self):
        t = self.make_tracker(); self.step(t, .9)
        for _ in range(8): self.step(t)
        self.assertEqual(len(t.tracked_stracks) + len(t.lost_stracks), 0)


if __name__ == '__main__': unittest.main()
