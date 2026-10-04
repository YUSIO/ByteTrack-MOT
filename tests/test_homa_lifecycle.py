"""Regression contracts for actual MHA lifecycle behavior and causal interventions."""
import unittest
from types import SimpleNamespace
import numpy as np
from homa.tracker import MHATracker
from homa.association import HOMAAssociation
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.basetrack import BaseTrack


def trace(kind, scores, **flags):
    BaseTrack._count=0
    args=SimpleNamespace(track_thresh=.6,track_buffer=30,match_thresh=.9,mot20=False,**flags)
    t=BYTETracker(args) if kind=='byte' else MHATracker(args)
    if kind!='byte': t.homa=HOMAAssociation(None,'mha_iou',window=8)
    result=[]
    for f,score in enumerate(scores,1):
        if kind!='byte':t.homa.set_current(None,f)
        det=np.empty((0,5),dtype=np.float32) if score is None else np.array([[0,0,20,20,score]],dtype=np.float32)
        out=t.update(det,(100,100),(100,100))
        result.append([x.track_id for x in out])
    return result


class LifecycleContracts(unittest.TestCase):
    def test_high_low_high_exposes_score_gate(self):
        self.assertEqual(trace('mha',[.9,.4,.9]),[[1],[1],[2]])
        self.assertEqual(trace('mha',[.9,.4,.9],mha_all_pool=True),[[1],[1],[1]])
        self.assertEqual(trace('byte',[.9,.4,.9]),[[1],[1],[1]])

    def test_gap_eight_expires_default_mha(self):
        seq=[.9]+[None]*7+[.9]
        self.assertEqual(trace('mha',seq)[-1],[2])
        self.assertEqual(trace('mha',seq,mha_max_gap=30)[-1],[1])
        self.assertEqual(trace('byte',seq)[-1],[1])

    def test_late_birth_confirmation_difference(self):
        self.assertEqual(trace('mha',[None,.9,.9]),[[],[1],[1]])
        self.assertEqual(trace('byte',[None,.9,.9]),[[],[],[1]])


if __name__=='__main__': unittest.main()
