import unittest
import numpy as np
from fctrack.weak_birth_gate import FEATURE_NAMES,features,predict
from test_weak_birth import make

class GateTests(unittest.TestCase):
    def test_features_geometric_invariants_and_extreme_logits(self):
        raw=np.array([[0,0,10,10],[3,2,10,10]],float)
        x=features([0,1],[1,0],raw,raw,[0,.1],[0,.1])
        self.assertTrue(np.isfinite(x).all());np.testing.assert_allclose(x[:,2],1)
        np.testing.assert_allclose(x[:,3:5],0)
        shifted=raw.copy();shifted[:,:2]+=100
        np.testing.assert_allclose(features([0,1],[1,0],shifted,shifted,[0,.1],[0,.1]),x)

    def test_predict_schema_and_known_scores(self):
        model=dict(feature_names=list(FEATURE_NAMES[:2]),mean=[0,0],std=[1,1],coef=[1,0],intercept=0)
        x=np.zeros((2,8));x[:,0]=[0,np.log(3)]
        np.testing.assert_allclose(predict(model,x),[.5,.75])
        model['std']=[0,1]
        with self.assertRaises(ValueError):predict(model,x)

    def test_runtime_is_gt_blind_and_replaces_only_acceptance(self):
        class NoGT(dict):
            def get(self,*args):raise AssertionError('GT read')
        f=make('learned');f.gt=NoGT();f.weak_birth.update(p_min=.97,use_regressed_box=True,threshold=.5,
          model=dict(feature_names=['det_logit','seed_logit'],mean=[0,0],std=[1,1],coef=[0,0],intercept=1))
        f.verify=lambda im,b:(np.zeros(len(b)),b.copy())
        raw=np.array([[10,10,20,20,.3]],np.float32);before=raw.copy()
        rows=f.step(raw,(100,100),lambda:None)
        self.assertEqual(len(rows),1);np.testing.assert_array_equal(rows[0][1],[10,10,10,10]);np.testing.assert_array_equal(raw,before)

if __name__=='__main__':unittest.main()
