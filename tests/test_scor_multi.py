import unittest
import numpy as np
from fctrack.scor_relabel import RelativeEdges

class MultiTests(unittest.TestCase):
    def setup_edge(self):
        e=RelativeEdges();ref={1:np.array([0.,0.]),2:np.array([10.,0.]),3:np.array([0.,20.])}
        e.predict(1);e.observe(ref,{k:(1,v,10) for k,v in ref.items()},300)
        return e,ref
    def test_translation_and_single_fallback(self):
        e,ref=self.setup_edge();e.predict(2);cur={k:ref[k]+[30.,40.] for k in (2,3)}
        p,ids,n,w,spread=e.estimate_local(1,cur,ref,ref[1],10)
        np.testing.assert_allclose(p,[30,40]);self.assertEqual(set(ids),{2,3});self.assertAlmostEqual(sum(w),1)
        p,ids,n,w,spread=e.estimate_local(1,{2:cur[2]},ref,ref[1],10)
        np.testing.assert_allclose(p,[30,40]);self.assertEqual(w,[1.])
        p,ids,n,w,spread=e.estimate_local(1,{},ref,ref[1],10)
        np.testing.assert_array_equal(p,ref[1]);self.assertEqual(n,0)
    def test_relative_velocity_and_distance_weights(self):
        e,ref=self.setup_edge()
        e.edges[1,2][0][2:]=[2,0];e.edges[1,3][0][2:]=[0,0];e.predict(4)
        cur={2:ref[2]+[30.,40.],3:ref[3]+[30.,40.]}
        p,ids,n,w,spread=e.estimate_local(1,cur,ref,ref[1],10)
        self.assertEqual(spread,6);np.testing.assert_allclose(p,[34,40]);self.assertEqual(n,2)
        p,ids,n,w,spread=e.estimate_local(1,cur,ref,ref[1],10,multi=False)
        np.testing.assert_allclose(p,[30,40]);self.assertEqual(ids,[3])
    def test_current_birth_without_historical_edge_excluded(self):
        e,ref=self.setup_edge();cur={2:np.array([40.,40.]),4:np.array([0.,0.])}
        ref[4]=np.array([1.,0.])
        p,ids,n,w,spread=e.estimate_local(1,cur,ref,ref[1],10)
        self.assertEqual(ids,[2]);np.testing.assert_allclose(p,[30,40])

if __name__=='__main__':unittest.main()
