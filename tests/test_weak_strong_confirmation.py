import unittest
import numpy as np
from test_weak_birth import make

EMPTY = np.empty((0, 5), np.float32)

def det(score):
    return np.array([[10, 10, 20, 20, score]], np.float32)

def configured(enabled=True):
    f = make('verifier')
    f.weak_birth.update(p_min=.97, require_strong_detection_for_recovery=enabled)
    f.rec = dict(estimator='kalman', lookback=5, max_since_real=30, min_observations=1,
                 iou_free=.3, skip_if_detected=False, p_min=.97, use_regressed_box=True, score=.7)
    return f

class StrongConfirmationTests(unittest.TestCase):
    def step(self, f, d):
        return f.step(d, (100, 100), lambda: None)

    def test_immediate_birth_and_weak_continuation_do_not_license_recovery(self):
        f = configured()
        a = self.step(f, det(.3)); self.assertEqual(len(a), 1)
        b = self.step(f, det(.3)); self.assertEqual(a[0][0], b[0][0])
        self.assertEqual(self.step(f, EMPTY), [])
        self.assertEqual(f.stats['fill_in_skipped_weak_strong_confirmation'], 1)
        self.assertEqual(f.stats['weak_recovery_strong_confirmed'], 0)

    def test_strong_real_update_licenses_recovery_and_preserves_score(self):
        f = configured(); a = self.step(f, det(.3))
        b = self.step(f, det(.8)); self.assertAlmostEqual(b[0][2], .8, places=6)
        c = self.step(f, EMPTY); self.assertEqual(c[0][0], a[0][0])
        self.assertEqual(f.stats['weak_recovery_strong_confirmed'], 1)

    def test_native_strong_track_is_unrestricted(self):
        f = configured(); a = self.step(f, det(.99)); b = self.step(f, EMPTY)
        self.assertEqual(a[0][0], b[0][0]); self.assertEqual(f.stats['weak_born'], 0)

    def test_disabled_recovers_without_detection_confirmation(self):
        f = configured(False); a = self.step(f, det(.3)); b = self.step(f, EMPTY)
        self.assertEqual(a[0][0], b[0][0]); self.assertEqual(f.stats['weak_recovery_strong_confirmed'], 0)

    def test_relabel_to_old_strong_identity_does_not_transfer_license(self):
        f = configured(); self.step(f, EMPTY)
        f.relab = dict(family='form', tau=6, gap=300)
        f.lastw[100] = (1, np.array([15., 15.]), 10.)
        f.hist[1] = {100: np.array([15., 15.])}
        f.meta[100] = dict(obs=20, real=1, box=np.array([10., 10., 10., 10.]), prov=0)
        a = self.step(f, det(.3)); self.assertEqual(a[0][0], 100)
        self.assertEqual(f.stats['relabelled'], 1)
        self.assertTrue(f.t.tracked_stracks[0]._weak_recovery_pending)
        self.assertEqual(self.step(f, EMPTY), [])

    def test_repeated_birth_is_new_pending_object(self):
        f = configured(); self.step(f, det(.3)); original = f.t.tracked_stracks[0]
        self.step(f, EMPTY); self.step(f, det(.3))
        new = f.t.tracked_stracks[0]
        self.assertIsNot(original, new); self.assertTrue(new._weak_recovery_pending)
        self.assertEqual(f.stats['weak_born'], 2)

    def test_deployable_path_has_no_gt_reads(self):
        class NoGT(dict):
            def get(self, *a): raise AssertionError('GT read')
        f = configured(); f.gt = NoGT()
        for d in [det(.3), det(.3), EMPTY, det(.8), EMPTY]: self.step(f, d)

    def test_equal_threshold_is_not_a_high_detection(self):
        f = configured(); self.step(f, det(.3)); self.step(f, det(.6))
        self.assertEqual(f.stats['weak_recovery_strong_confirmed'], 0)

if __name__ == '__main__': unittest.main()
