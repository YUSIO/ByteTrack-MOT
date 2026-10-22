"""Analytical regression cases derived from paper equations, not tracker outputs."""
import math
import unittest
import numpy as np
from yolox.tracker import matching
from yolox.tracker.byte_tracker import build_local_topologies, local_topology_cost, STrack, topology_fused_cost

class PaperEquations(unittest.TestCase):
    def test_global_distance_scale_and_translation(self):
        boxes=np.array([[0,0,2,2],[3,0,5,2],[7,0,9,2]], dtype=float)
        d=build_local_topologies(boxes)
        np.testing.assert_allclose(d[0][:,0], [9/14,21/14],rtol=1e-6)
        for a,b in zip(d,build_local_topologies(boxes*3+100)):
            np.testing.assert_allclose(a,b,rtol=1e-6)
    def test_direction_and_count_penalty(self):
        a=np.array([[1,0,1],[1,0,1]],dtype=float)
        b=np.array([[1,1,0],[1,1,0]],dtype=float)
        self.assertAlmostEqual(local_topology_cost(a,b),.4*math.sqrt(2)/2,places=6)
        self.assertAlmostEqual(local_topology_cost(a,np.vstack([a,a[:1]])),.1,places=6)
        self.assertAlmostEqual(local_topology_cost(a,np.array([[100,0,1]]*2)),1.,places=6)
        self.assertAlmostEqual(local_topology_cost(a,a[::-1]),0.,places=6)
    def test_fallback_and_refresh(self):
        self.assertEqual(build_local_topologies([[0,0,2,2],[5,0,7,2]]),[None,None])
        a=STrack([0,0,2,2],.9,topology=np.ones((2,3)))
        a.update_topology(None,3)
        self.assertIsNone(a.topology)
        costs=np.array([[.7]])
        np.testing.assert_allclose(topology_fused_cost(costs,[a],[a]),costs)
    def test_same_iou_convention(self):
        a=np.array([[0,0,2,2]],float); b=np.array([[1,0,3,2]],float)
        # ByteTrack inclusive IoU=6/12=.5; geometric penalty=1/(3^2+2^2).
        self.assertAlmostEqual(matching.dious(a,b)[0,0],.5-1/13)
        self.assertAlmostEqual(matching.dious(a,b,True)[0,0],1/3-1/13)
        self.assertEqual(matching.dious(a,[]).shape,(1,0))

if __name__=='__main__': unittest.main()
