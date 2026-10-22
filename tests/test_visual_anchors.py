import unittest
from types import SimpleNamespace
import numpy as np
from fctrack.tracker import FCTrack, wrap_stracks
from yolox.tracker.byte_tracker import BYTETracker, STrack
from yolox.tracker.basetrack import BaseTrack


class AnchorTests(unittest.TestCase):
    def make(self, birth=False):
        wrap_stracks(STrack)
        BaseTrack._count = 0
        tracker = BYTETracker(SimpleNamespace(track_thresh=.6,track_buffer=30,match_thresh=.9,mot20=False))
        cfg = dict(recover=dict(estimator='cotracker|kalman',lookback=5,max_since_real=30,
                   min_observations=1,iou_free=.3,skip_if_detected=False,p_min=.97,
                   use_regressed_box=True,score=.7))
        if birth:
            cfg['birth']=dict(lo=.6,hi=1.01,cover_iou=.3,rule='verifier',p_min=.97,use_regressed_box=True)
        fc=FCTrack(tracker,STrack,cfg,lambda image,b:(np.ones(len(b)),b.copy()))
        calls=[]
        def provider(frame,t):
            calls.append((frame,getattr(t,'_visual_anchor_frame',None),getattr(t,'_visual_anchor_box',None)))
            if not hasattr(t,'_visual_anchor_frame'):
                return None
            return np.array([15.+frame,15.])
        fc.point_provider=provider
        return fc,calls

    def step(self,fc,x=None,score=.9):
        d=np.array([[x,10,x+10,20,score]],np.float32) if x is not None else np.empty((0,5),np.float32)
        return fc.step(d,(100,100),lambda:np.zeros((100,100,3),np.uint8))

    def test_recovery_does_not_replace_real_anchor(self):
        fc,calls=self.make()
        self.step(fc,10)
        self.step(fc)
        self.step(fc)
        self.assertEqual([x[1] for x in calls],[1,1])
        for _,_,box in calls:
            np.testing.assert_array_equal(box,[10,10,10,10])
        self.assertNotEqual(fc.meta[1]['box'][0],10)
        self.step(fc,13)
        self.step(fc)
        self.assertEqual(calls[-1][1],4)
        np.testing.assert_array_equal(calls[-1][2],[13,10,10,10])

    def test_verifier_birth_waits_for_detector_update_to_anchor(self):
        fc,calls=self.make(True)
        self.step(fc)
        rows=self.step(fc,10,.65)
        self.assertEqual(len(rows),1)
        track=fc.t.tracked_stracks[0]
        self.assertFalse(hasattr(track,'_visual_anchor_frame'))
        self.step(fc,10,.9)
        self.assertEqual(track._visual_anchor_frame,3)


if __name__=='__main__':
    unittest.main()
