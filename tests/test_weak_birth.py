import unittest
from types import SimpleNamespace
import numpy as np
from fctrack.tracker import FCTrack, wrap_stracks
from yolox.tracker.byte_tracker import BYTETracker, STrack
from yolox.tracker.basetrack import BaseTrack


def make(weak=None):
    BaseTrack._count = 0
    wrap_stracks(STrack)
    t = BYTETracker(SimpleNamespace(track_thresh=.6, track_buffer=30, match_thresh=.9, mot20=False))
    cfg = {'birth': dict(lo=.6, hi=1.01, cover_iou=.3, rule='verifier', p_min=.97, use_regressed_box=True)}
    if weak is not None:
        cfg['weak_birth'] = dict(lo=.1, hi=.6, cover_iou=.3, rule=weak, use_regressed_box=False)
    return FCTrack(t, STrack, cfg, lambda im,b: (np.ones(len(b)), b.copy()))


class WeakBirthTests(unittest.TestCase):
    def test_disabled_keeps_low_out(self):
        f=make(); d=np.array([[10,10,20,20,.3]],np.float32)
        self.assertEqual(f.step(d,(100,100),lambda:None),[])

    def test_oracle_preserves_raw_box_score_and_new_id(self):
        f=make('oracle'); f.gt={1:np.array([[10,10,10,10]])}
        d=np.array([[10,10,20,20,.3],[50,50,60,60,.4]],np.float32); before=d.copy()
        rows=f.step(d,(100,100),lambda:None)
        self.assertEqual(len(rows),1); self.assertEqual(rows[0][0],1)
        np.testing.assert_array_equal(rows[0][1],[10,10,10,10]); np.testing.assert_array_equal(d,before)
        self.assertAlmostEqual(rows[0][2],.3,places=6)
        self.assertEqual(f.weak_birth_log[0]['public_id'],1)
        self.assertEqual(f.bth['lo'],.6)

    def test_oracle_no_gt_and_duplicate_coverage(self):
        f=make('oracle');d=np.array([[10,10,20,20,.3]],np.float32)
        self.assertEqual(f.step(d,(100,100),lambda:None),[])
        f=make('oracle');f.gt={1:np.array([[10,10,10,10]])}
        rows=f.step(np.array([[10,10,20,20,.99],[10,10,20,20,.3]],np.float32),(100,100),lambda:None)
        self.assertEqual(len(rows),1);self.assertEqual(f.stats['weak_born'],0)

    def test_strong_birth_first_and_bands_disjoint(self):
        f=make('oracle');f.gt={1:np.array([[10,10,10,10],[40,40,10,10]])}
        rows=f.step(np.array([[10,10,20,20,.65],[40,40,50,50,.3]],np.float32),(100,100),lambda:None)
        self.assertEqual(len(rows),2); self.assertEqual(f.stats['weak_born'],1)
        with self.assertRaises(ValueError):
            FCTrack(None,None,{'birth':{'lo':.6},'weak_birth':{'lo':.1,'hi':.7}})

    def test_weak_birth_can_continue_without_rescoring(self):
        f=make('oracle');f.gt={1:np.array([[10,10,10,10]])}
        d=np.array([[10,10,20,20,.3]],np.float32)
        a=f.step(d,(100,100),lambda:None);b=f.step(d,(100,100),lambda:None)
        self.assertEqual(a[0][0],b[0][0]);self.assertEqual(f.stats['weak_born'],1)

    def test_deployable_verifier_ignores_gt_and_applies_threshold(self):
        class NoGT(dict):
            def get(self,*args): raise AssertionError('deployable birth read GT')
        f=make('verifier');f.gt=NoGT();f.weak_birth['p_min']=.97
        f.weak_birth['use_regressed_box']=True
        f.verify=lambda im,b:(np.array([.98,.96]),b+np.array([1,0,0,0]))
        d=np.array([[10,10,20,20,.3],[50,50,60,60,.4]],np.float32)
        rows=f.step(d,(100,100),lambda:None)
        self.assertEqual(len(rows),1);np.testing.assert_array_equal(rows[0][1],[11,10,10,10])

    def test_observation_is_non_mutating(self):
        f=make('verifier');f.weak_birth.update(p_min=.97,use_regressed_box=True,log_queries=True,observe_only=True)
        d=np.array([[10,10,20,20,.3]],np.float32)
        self.assertEqual(f.step(d,(100,100),lambda:None),[])
        self.assertEqual(f.stats['weak_born'],0);self.assertEqual(len(f.weak_query_log),1)
        self.assertEqual(f.t.tracked_stracks,[])

    def test_regressed_oracle_keeps_model_box_ignores_p_and_not_gt_coordinates(self):
        f=make('oracle_regressed');f.weak_birth.update(p_min=.97,use_regressed_box=True)
        f.gt={1:np.array([[12,10,10,10]])}
        f.verify=lambda im,b:(np.zeros(len(b)),np.array([[11.,10.,10.,10.]]))
        rows=f.step(np.array([[0,0,5,5,.3]],np.float32),(100,100),lambda:None)
        self.assertEqual(len(rows),1);np.testing.assert_array_equal(rows[0][1],[11,10,10,10])
        self.assertAlmostEqual(rows[0][2],.3,places=6)


if __name__=='__main__': unittest.main()
