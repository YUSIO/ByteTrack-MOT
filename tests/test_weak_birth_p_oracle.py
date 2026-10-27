"""Diagnostic oracle scope: fixed p gate never supplies GT coordinates or IDs."""
import unittest
import numpy as np
from test_weak_birth import make

class WeakPOracleTests(unittest.TestCase):
    def setup(self, required, p, gt=True):
        f=make('oracle_regressed')
        f.weak_birth.update(use_regressed_box=True,p_min=.97,oracle_require_p=required)
        f.verify=lambda im,b:(np.full(len(b),p),np.tile([11.,10.,10.,10.],(len(b),1)))
        if gt:f.gt={1:np.array([[12.,10.,10.,10.]])}
        return f

    def step(self,f):
        return f.step(np.array([[0.,0.,5.,5.,.3]],np.float32),(100,100),lambda:None)

    def test_original_oracle_ignores_low_p(self):
        f=self.setup(False,.1);a=self.step(f)
        self.assertEqual(len(a),1);np.testing.assert_array_equal(a[0][1],[11,10,10,10])

    def test_restricted_oracle_rejects_low_p_even_with_correct_box(self):
        self.assertEqual(self.step(self.setup(True,.96)),[])

    def test_restricted_oracle_accepts_exact_boundary_and_keeps_model_box(self):
        a=self.step(self.setup(True,.97));self.assertEqual(len(a),1)
        np.testing.assert_array_equal(a[0][1],[11,10,10,10])
        self.assertAlmostEqual(a[0][2],.3,places=6)

    def test_high_p_still_requires_gt_match(self):
        self.assertEqual(self.step(self.setup(True,.99,gt=False)),[])

    def test_current_gt_already_covered_is_not_born_twice(self):
        f=self.setup(True,.99);f.gt={1:np.array([[10.,10.,10.,10.]])}
        d=np.array([[10.,10.,20.,20.,.99],[0.,0.,5.,5.,.3]],np.float32)
        a=f.step(d,(100,100),lambda:None)
        self.assertEqual(len(a),1);self.assertEqual(f.stats['weak_born'],0)

if __name__=='__main__':unittest.main()
