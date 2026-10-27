import unittest
from types import SimpleNamespace
import numpy as np
from fctrack.tracker import FCTrack,wrap_stracks
from yolox.tracker.byte_tracker import BYTETracker,STrack
from yolox.tracker.basetrack import BaseTrack
class ForbiddenGT(dict):
    def get(self,*args): raise AssertionError('deployable gate read GT')
def make(rule='learned',allow_weak=True):
    BaseTrack._count=0;wrap_stracks(STrack)
    tracker=BYTETracker(SimpleNamespace(track_thresh=.6,track_buffer=30,match_thresh=.9,mot20=False))
    fc=FCTrack(tracker,STrack,{'unified_birth':dict(rule=rule,score_lower=.1,cover_iou=.3,allow_weak=allow_weak,threshold=.8,p_min=.97)},lambda im,b:(np.ones(len(b)),b.copy()))
    fc.unified_predict=lambda frame,ids,boxes,image:np.ones(len(ids))
    return fc
def det(score=.3,x=10):return np.array([[x,10,x+10,20,score]],np.float32)
class UnifiedBirthTests(unittest.TestCase):
    def test_native_high_cannot_bypass_rejected_gate(self):
        fc=make();fc.unified_predict=lambda *a:np.zeros(len(a[1]));fc.gt=ForbiddenGT()
        self.assertEqual(fc.step(det(.99),(100,100),lambda:None),[]);self.assertEqual(fc.t.tracked_stracks,[])
    def test_weak_immediate_raw_score_and_continuation(self):
        fc=make();fc.gt=ForbiddenGT();d=det();before=d.copy()
        a=fc.step(d,(100,100),lambda:None);b=fc.step(d,(100,100),lambda:None)
        self.assertEqual(a[0][0],b[0][0]);self.assertEqual(fc.stats['weak_born'],1)
        np.testing.assert_array_equal(d,before);np.testing.assert_array_equal(a[0][1],[10,10,10,10])
        self.assertEqual(a[0][2],d[0,4]);self.assertEqual(fc.t.used_detection_indices,{0})
    def test_strict_lower_and_exact_point_six(self):
        fc=make();rows=fc.step(np.vstack([det(.1),det(.6,40)]),(100,100),lambda:None)
        self.assertEqual(len(rows),1);self.assertEqual(fc.stats['unified_strong_born'],1);self.assertEqual(fc.unified_query_log[0]['detection_index'],1)
    def test_final_overlap_and_score_order(self):
        fc=make();d=np.vstack([det(.3),det(.5)]);rows=fc.step(d,(100,100),lambda:None)
        self.assertEqual(len(rows),1);self.assertEqual(rows[0][2],d[1,4]);self.assertEqual(fc.stats['unified_final_overlap'],1)
    def test_oracle_selects_raw_not_GT_and_weak_off(self):
        fc=make('oracle');fc.gt={1:np.array([[11,10,10,10]])}
        rows=fc.step(np.vstack([det(.3),det(.99,60)]),(100,100),lambda:None)
        self.assertEqual(len(rows),1);np.testing.assert_array_equal(rows[0][1],[10,10,10,10])
        fc=make('oracle',False);fc.gt={1:np.array([[10,10,10,10]])};self.assertEqual(fc.step(det(.3),(100,100),lambda:None),[])
    def test_oracle_one_to_one_no_duplicate_and_no_GT(self):
        fc=make('oracle');fc.gt={1:np.array([[10,10,10,10]])}
        self.assertEqual(len(fc.step(np.vstack([det(.3),det(.5)]),(100,100),lambda:None)),1)
        fc=make('oracle');fc.gt={};self.assertEqual(fc.step(det(.99),(100,100),lambda:None),[])
    def test_existing_associated_detection_never_queries_new_gate(self):
        fc=make();fc.step(det(.99),(100,100),lambda:None)
        def forbidden(*a):self.assertEqual(len(a[1]),0);return np.empty(0)
        fc.unified_predict=forbidden;self.assertEqual(len(fc.step(det(.99),(100,100),lambda:None)),1);self.assertEqual(fc.stats['unified_born'],1)
    def test_bridge_never_weak_no_regression_output(self):
        fc=make('bridge');fc.verify=lambda im,b:(np.ones(len(b)),b+100)
        rows=fc.step(np.vstack([det(.3),det(.65,40),det(.9,70)]),(100,100),lambda:None)
        self.assertEqual(len(rows),2);self.assertEqual(fc.stats['weak_born'],0);self.assertTrue(any(np.array_equal(r[1],[40,10,10,10]) for r in rows))
    def test_old_birth_path_excluded(self):
        fc=make()
        with self.assertRaises(ValueError):FCTrack(fc.t,STrack,{'unified_birth':{},'birth':{'lo':.6}})
    def test_same_score_deterministic_index_and_input_unchanged(self):
        fc=make();d=np.vstack([det(.5),det(.5)]);before=d.copy();fc.step(d,(100,100),lambda:None)
        np.testing.assert_array_equal(d,before);self.assertEqual(fc.unified_birth_log[0]['detection_index'],0)
if __name__=='__main__':unittest.main()
