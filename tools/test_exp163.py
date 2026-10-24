"""Mechanism and evaluator regression tests; no dataset or GT in tracker."""
import unittest
from types import SimpleNamespace
from collections import defaultdict
import numpy as np
import motmetrics as mm
from yolox.tracker.byte_tracker import STrack
from yolox.tracker.basetrack import TrackState
from yolox.tracker.swarm_residual import SwarmResidual
from evaluate_uavswarm_mot import strict_mot_accumulator


def state(x, y, kind=TrackState.Tracked):
    return SimpleNamespace(mean=np.array([x,y,1,10,2,1,0,0.],float),
                           covariance=np.eye(8), state=kind, is_activated=True, end_frame=1)


class Regression(unittest.TestCase):
    def test_zero_innovation_after_nontrivial_gmc(self):
        tracks=[state(20,30),state(60,50)]
        STrack.multi_gmc(tracks,np.array([[0.,-1.,100.],[1.,0.,20.]]))
        dets=[STrack([t.mean[0]-5,t.mean[1]-5,10,10],.9) for t in tracks]
        r=SwarmResidual(200,200); r.collect(tracks,dets,[(0,0),(1,1)],2)
        np.testing.assert_array_equal(r.shift,[0,0])

    def test_only_center_changes_and_gmc_is_not_bypassed(self):
        anchors=[state(20,30),state(60,50)]
        dets=[STrack([t.mean[0]-2,t.mean[1]-7,10,10],.9) for t in anchors]
        r=SwarmResidual(200,200); r.collect(anchors,dets,[(0,0),(1,1)],2)
        lost=state(80,80,TrackState.Lost); before=lost.mean.copy(); cov=lost.covariance.copy()
        r.correct([lost],2)
        np.testing.assert_array_equal(lost.mean[:2],before[:2]+[3,-2])
        np.testing.assert_array_equal(lost.mean[2:],before[2:])
        np.testing.assert_array_equal(lost.covariance,cov)
        STrack.multi_gmc([lost],np.array([[1.,0.,7.],[0.,1.,4.]]))
        np.testing.assert_array_equal(lost.mean[:2],[90,82])

    def test_no_anchors_shadow_ttl_and_tracked(self):
        r=SwarmResidual(200,200); lost=state(80,80,TrackState.Lost); before=lost.mean.copy()
        r.collect([],[],[],2); r.correct([lost],2)
        np.testing.assert_array_equal(lost.mean,before)
        r.shift=np.array([3.,2.]); r.mode='shadow'; r.correct([lost],2)
        np.testing.assert_array_equal(lost.mean,before)
        r.mode='apply'; r.correct([lost],32)
        np.testing.assert_array_equal(lost.mean,before)
        lost.state=TrackState.Tracked; r.correct([lost],2)
        np.testing.assert_array_equal(lost.mean,before)

    def test_dispersion_gate_and_lost_anchor_exclusion(self):
        a=[state(20,30),state(60,50,TrackState.Lost)]
        d=[STrack([15,25,10,10],.9),STrack([95,95,10,10],.9)]
        r=SwarmResidual(200,200); r.collect(a,d,[(0,0),(1,1)],2)
        self.assertIsNone(r.shift)
        a[1].state=TrackState.Tracked; r.collect(a,d,[(0,0),(1,1)],2)
        self.assertIsNone(r.shift)

    def test_prediction_only_frame_and_empty_prediction(self):
        g=defaultdict(list,{1:[(1,0,0,10,10)],3:[(1,0,0,10,10)]})
        p=defaultdict(list,{1:[(1,0,0,10,10)],2:[(1,0,0,10,10)]})
        result=mm.metrics.create().compute(strict_mot_accumulator(g,p),
            metrics=['num_false_positives','num_misses','num_switches'])
        self.assertEqual(result.iloc[0]['num_false_positives'],1)
        self.assertEqual(result.iloc[0]['num_misses'],1)
        self.assertEqual(result.iloc[0]['num_switches'],0)
        empty=mm.metrics.create().compute(strict_mot_accumulator(g,{}),metrics=['num_misses'])
        self.assertEqual(empty.iloc[0]['num_misses'],2)


if __name__ == '__main__':
    unittest.main()
