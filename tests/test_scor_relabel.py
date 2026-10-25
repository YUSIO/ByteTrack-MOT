import unittest
from types import SimpleNamespace
import numpy as np
from fctrack.scor_relabel import RelativeEdges, ScorRelabelTrack

class ScorTests(unittest.TestCase):
    def test_relative_cv_predicts_gap_not_common_translation(self):
        e=RelativeEdges();last={}
        for f in range(1,31):
            p={1:np.array([3*f+f,2*f+10.]),2:np.array([3*f,2*f])}
            e.predict(f);last={i:(f,v,1) for i,v in p.items()};e.observe(p,last,300)
        e.predict(35)
        pred,j,n=e.estimate(1,{2:np.array([105.,70.])},{2:np.array([90.,60.])},p[1])
        np.testing.assert_allclose(pred,[140,80],atol=.02)
        self.assertEqual(j,2)
        np.testing.assert_allclose(e.edges[1,2][1],e.edges[1,2][1].T)
        self.assertGreaterEqual(np.linalg.eigvalsh(e.edges[1,2][1]).min(),0)
    def test_common_translation_and_no_anchor(self):
        e=RelativeEdges();p={1:np.array([10.,5]),2:np.array([0.,0])}
        e.predict(1);e.observe(p,{i:(1,v,1) for i,v in p.items()},300);e.predict(2)
        pred,j,n=e.estimate(1,{2:np.array([40.,20.])},p,p[1])
        np.testing.assert_allclose(pred,[50,25])
        pred,j,n=e.estimate(1,{},p,p[1]);np.testing.assert_array_equal(pred,p[1]);self.assertIsNone(j)
    def tracker(self,gate='distance',gap=300):
        t=SimpleNamespace(lost_stracks=[SimpleNamespace(track_id=1)],removed_stracks=[])
        fc=ScorRelabelTrack(t,None,{'relabel':dict(family='abs',tau=6,gap=gap,gate=gate)})
        fc.lastw={1:(1,np.array([5.,5.]),10.)};fc.hist={1:{1:np.array([5.,5.])}};fc.lastbox={1:np.array([0.,0.,10.,10.])}
        return fc
    def test_zero_iou_gate_and_new_state_preservation(self):
        fc=self.tracker();obj=SimpleNamespace(track_id=2,mean=np.array([123.]),covariance=np.eye(1))
        rows=[[2,np.array([20.,0,10.,10.]),.9,obj]]
        fc.relabel_births(3,rows)
        self.assertEqual(rows[0][0],1);np.testing.assert_array_equal(obj.mean,[123]);self.assertFalse(fc.t.lost_stracks)
        fc=self.tracker('overlap');rows=[[2,np.array([20.,0,10.,10.]),.9,SimpleNamespace(track_id=2)]]
        fc.relabel_births(3,rows);self.assertEqual(rows[0][0],2)
    def test_expiry_and_one_to_one(self):
        fc=self.tracker(gap=30);rows=[[2,np.array([0.,0,10.,10.]),.9,SimpleNamespace(track_id=2)]]
        fc.relabel_births(32,rows);self.assertEqual(rows[0][0],2)
        fc=self.tracker();rows=[[i,np.array([float(i),0,10.,10.]),.9,SimpleNamespace(track_id=i)] for i in (2,3)]
        fc.relabel_births(3,rows);self.assertEqual(sum(r[0]==1 for r in rows),1)

if __name__=='__main__':unittest.main()
