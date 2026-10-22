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

    def test_consistent_coordinate_geometry(self):
        a=np.array([[0,0,2,2]],float); b=np.array([[1,0,3,2]],float)
        self.assertAlmostEqual(matching.dious(a,b,geometry="pixel")[0,0],.5-1/25)
        self.assertAlmostEqual(matching.dious(a,b,geometry="continuous")[0,0],1/3-1/13)
        # Inclusive endpoints equal expanding each box by half a pixel on each side.
        expand=np.array([-.5,-.5,.5,.5])
        np.testing.assert_allclose(matching.dious(a,b,geometry="pixel"), matching.dious(a+expand,b+expand,geometry="continuous"))
        np.testing.assert_allclose(matching.iou_distance(a,b,geometry="continuous"),[[2/3]])
        np.testing.assert_allclose(matching.iou_distance(a,b,geometry="pixel"),[[.5]])

    def test_score_fusion_and_gate(self):
        from types import SimpleNamespace
        detections=[SimpleNamespace(score=.8)]
        # IoU .2, penalty .1: IoU cost .84 survives .9; DIoU cost .92 fails.
        iou=matching.fuse_score(np.array([[.8]]),detections)
        diou=matching.fuse_score(np.array([[.9]]),detections)
        np.testing.assert_allclose(iou,[[.84]])
        np.testing.assert_allclose(diou,[[.92]])
        self.assertEqual(len(matching.linear_assignment(iou,.9)[0]),1)
        self.assertEqual(len(matching.linear_assignment(diou,.9)[0]),0)
        self.assertEqual(len(matching.linear_assignment(diou,.99)[0]),1)
        # Negative DIoU stays negative in similarity; no hidden clipping or /2.
        self.assertAlmostEqual(matching.fuse_score(np.array([[1.3]]),detections)[0,0],1.24)
        # LAPJV jointly optimizes matching and unmatched penalties, not Hungarian+filter.
        cost=np.array([[.1,.8],[.8,1.1]])
        matched=matching.linear_assignment(cost,.9)[0]
        self.assertEqual(set(map(tuple,matched)),{(0,0)})

    def test_shadow_cost_order(self):
        from types import SimpleNamespace
        rng=np.random.default_rng(143)
        xy=rng.uniform(-20,20,(20,2)); wh=rng.uniform(.1,30,(20,2))
        boxes=np.column_stack([xy,xy+wh]); dets=[SimpleNamespace(score=s) for s in rng.uniform(.6,1,20)]
        for geometry in ("hybrid","continuous","pixel"):
            a=matching.iou_distance(boxes,boxes[::-1],geometry=geometry)
            b=matching.diou_distance(boxes,boxes[::-1],geometry=geometry)
            self.assertTrue(np.all(b>=a-1e-12))
            self.assertTrue(np.all(matching.fuse_score(b,dets)>=matching.fuse_score(a,dets)-1e-12))

if __name__=='__main__': unittest.main()
