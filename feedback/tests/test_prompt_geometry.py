import unittest
import torch
from feedback.prompt import PriorPrompt

class GeometryTests(unittest.TestCase):
    def test_single_object_support_does_not_cancel(self):
        m=PriorPrompt(dim=8,backprojection="peak_coverage")
        a=torch.tensor([[.1,.2,.3,.4]]);p=torch.ones(1,8)
        torch.testing.assert_close(m.backproject(a,p),a.T/.4@p)
        self.assertFalse(torch.equal(m.backproject(a,p),m.backproject(a.flip(1),p)))

    def test_convex_bound_and_uniform_scale(self):
        m=PriorPrompt(dim=8,backprojection="peak_coverage")
        for n in (1,3):
            p=torch.randn(n,8)
            for count in (20,4500):
                a=torch.randn(n,count).softmax(-1);g=m.backproject(a,p)
                self.assertLessEqual(float(g.abs().max()),float(p.abs().max())+1e-6)
                a=torch.ones(n,count)/count
                torch.testing.assert_close(m.backproject(a,p),p.mean(0).expand(count,-1))

    def test_geometry_continuity_and_boundary(self):
        m=PriorPrompt(position_encoding=True,content_score="cosine",spatial_score="box_gaussian")
        boxes=torch.tensor([[0.,0.,16.,16.],[16.,0.,16.,16.]])
        patches=torch.zeros(20,64);q=m.encode_query(boxes,torch.tensor([1,1]),64,80)
        a=m.attention(q,patches,64,80,boxes)
        self.assertEqual(a.argmax(-1).tolist(),[0,1])
        self.assertGreater(float(a[0,0]),float(a[0,4]))
        g=m.geometry_logits(boxes,64,80,patches)
        shifted=m.geometry_logits(boxes+torch.tensor([.001,0.,0.,0.]),64,80,patches)
        self.assertLess(float((g-shifted).abs().max()),.001)

if __name__=='__main__':unittest.main()
