import unittest
import numpy as np
from types import SimpleNamespace
from collections import Counter
from fctrack.relabel_reliability import form_spread
from fctrack.scor_relabel import ScorRelabelTrack

class ReliabilityTests(unittest.TestCase):
    def test_shared_translation_and_missing_evidence(self):
        ref={1:np.array([0.,0.]),2:np.array([10.,0.]),3:np.array([0.,10.])}
        self.assertEqual(form_spread(1,ref[1],10,ref,{2:ref[2]+5})[0],None)
        self.assertEqual(form_spread(1,ref[1],10,ref,{})[0],None)
        v,n=form_spread(1,ref[1],10,ref,{j:ref[j]+[5,8] for j in [2,3]})
        self.assertAlmostEqual(v,0);self.assertEqual(n,2)
    def test_veto_preserves_identity_and_state(self):
        fc=ScorRelabelTrack.__new__(ScorRelabelTrack)
        fc.relab={'gap':300,'tau':6.,'family':'form','spread_veto_threshold':.1}
        fc.lastw={1:(1,np.array([0.,0.]),10.),2:(1,np.array([10.,0.]),10.),3:(1,np.array([0.,10.]),10.)}
        fc.hist={1:{i:x[1] for i,x in fc.lastw.items()}};fc.reliability_history=fc.hist
        fc.lastbox={i:np.r_[x[1]-5,[10.,10.]] for i,x in fc.lastw.items()}
        fc.t=SimpleNamespace(lost_stracks=[SimpleNamespace(track_id=1)],removed_stracks=[])
        fc.stats=Counter();fc.relabel_events=[];fc.meta={};fc.snap={}
        fc.prediction=lambda family,c1,side1,f1,f,L,cur:(c1,None,2)
        obj=SimpleNamespace(track_id=9)
        rows=[[9,np.array([-5.,-5.,10.,10.]),.9,obj], [2,np.array([5.,-5.,10.,10.]),.9,None], [3,np.array([15.,5.,10.,10.]),.9,None]]
        fc.relabel_births(2,rows)
        self.assertEqual(obj.track_id,9);self.assertEqual(rows[0][0],9)
        self.assertEqual(fc.t.lost_stracks[0].track_id,1)
        self.assertTrue(fc.relabel_events[0]['reliability_rejected'])
        self.assertEqual(fc.relabel_events[0]['pre_veto_chosen'],1)
        self.assertIsNone(fc.relabel_events[0]['chosen'])

if __name__=='__main__':unittest.main()
