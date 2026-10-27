import unittest
import numpy as np
from fctrack.spatial_readout import region_weights,regional_means

class SpatialReadoutTests(unittest.TestCase):
    def test_area_matches_projected_box(self):
        raw=np.array([[10.,20.,12.,12.]]);w=region_weights(raw,raw)
        self.assertAlmostEqual(w.sum()*256,(64/3)**2,places=4)
        self.assertTrue(((w>=0)&(w<=1)).all())

    def test_exact_grid_cell_and_full_window(self):
        # side=64/3 -> crop scale=1 and origin=0 exactly.
        raw=np.array([[64/3,64/3,64/3,64/3]])
        w=region_weights(raw,np.array([[0.,0.,16.,16.]]))
        np.testing.assert_allclose(w[0],np.diag([1.,0,0,0]),atol=1e-12)
        full=region_weights(raw,np.array([[0.,0.,64.,64.]]))
        np.testing.assert_allclose(full,1,atol=1e-12)

    def test_outside_empty_and_constant_feature(self):
        raw=np.array([[10.,10.,10.,10.]])
        w=region_weights(raw,np.array([[1000.,1000.,10.,10.]]))
        np.testing.assert_array_equal(w,0)
        h=np.full((1,3,4,4),7.)
        np.testing.assert_array_equal(regional_means(h,w),[[0,0,0,7,7,7]])

    def test_roll_changes_region_but_preserves_global_statistics(self):
        h=np.zeros((1,2,4,4));h[:,:,0,0]=16
        w=np.zeros((1,4,4));w[:,0,0]=1
        a=regional_means(h,w);b=regional_means(h,w,roll=True)
        np.testing.assert_array_equal(a[0,:2],[16,16]);np.testing.assert_array_equal(b[0,:2],[0,0])
        rolled=np.roll(h,(2,2),axis=(2,3))
        np.testing.assert_array_equal(h.mean((2,3)),rolled.mean((2,3)))
        np.testing.assert_array_equal(h.max((2,3)),rolled.max((2,3)))

    def test_fractional_weights_weighted_average(self):
        h=np.arange(16,dtype=float).reshape(1,1,4,4)
        w=np.zeros((1,4,4));w[0,0,0]=.25;w[0,0,1]=.75
        self.assertEqual(regional_means(h,w)[0,0],.75)

if __name__=='__main__':unittest.main()
