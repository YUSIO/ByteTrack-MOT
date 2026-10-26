from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.probe_weak_birth_jitter import proposals,consistency

class JitterTests(unittest.TestCase):
    def test_offsets_and_fixed_shape(self):
        raw=np.array([[1.,2.,4.,9.]])
        q=proposals(raw,[[.25,0],[-.25,0],[0,.25],[0,-.25]])
        np.testing.assert_array_equal(q[0,0],raw[0])
        np.testing.assert_allclose(q[0,1:,:2],[[2.5,2],[-.5,2],[1,3.5],[1,.5]])
        np.testing.assert_array_equal(q[0,:,2:],np.tile([4,9],(5,1)))
        np.testing.assert_array_equal(raw,[[1,2,4,9]])

    def test_agreement_invalid_and_moved_regressions(self):
        boxes=np.tile([0.,0.,4.,4.],(3,5,1));p=np.full((3,5),.99)
        boxes[1,4,0]=4;boxes[2,1,2]=0
        pmin,imin,iou=consistency(p,boxes)
        np.testing.assert_allclose(imin,[1,0,0]);np.testing.assert_allclose(pmin,[.99,.99,0])
        np.testing.assert_array_equal(iou[0],np.ones(4))

if __name__=='__main__':unittest.main()
