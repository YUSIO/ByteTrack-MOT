import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from yolox.models.yolo_head import YOLOXHead
from audit.augmentation_assignment import AssignmentObserver,save_preview,size_record

class AuditTests(unittest.TestCase):
    @torch.no_grad()
    def test_native_assignment_and_conflict(self):
        torch.set_num_threads(2);torch.manual_seed(14)
        head=YOLOXHead(1,in_channels=[16,32,64]).eval();head.training=True
        features=[torch.randn(1,c,s,s) for c,s in [(16,8),(32,4),(64,2)]]
        labels=torch.tensor([[[0.,20.,20.,3.,3.],[0.,20.1,20.1,4.,4.],[0.,48.,48.,16.,16.]]])
        image=torch.randn(1,3,64,64);expected=head(features,labels,image)
        observer=AssignmentObserver(head);actual=head(features,labels,image)
        for a,b in zip(expected,actual):
            if torch.is_tensor(a):self.assertTrue(torch.equal(a,b))
            else:self.assertEqual(a,b)
        self.assertEqual(len(observer.result),3)
        self.assertEqual(sum(r['positive_count'] for r in observer.result),int(actual[-1]*3))
        self.assertTrue(all(r['dynamic_k']>=1 for r in observer.result))
        self.assertTrue(any(r['positive_count']==0 for r in observer.result))

    def test_visible_geometry_and_preview(self):
        result=size_record([-5,2,5,6],(20,20))
        self.assertEqual(result['width'],10);self.assertEqual(result['visible_fraction'],.5)
        with tempfile.TemporaryDirectory() as path:
            save_preview(Path(path),np.zeros((3,64,64),np.float32),np.array([[0,20,20,3,3,1]],np.float32),dict(case_id=1,input_hw=[64,64]),dict(affine_scale=.2))
            self.assertTrue((Path(path)/'previews/case_0001.jpg').exists())

if __name__=='__main__':unittest.main()
