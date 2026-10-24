import unittest
import torch
from feedback.prompt import PriorPrompt

class BridgeTests(unittest.TestCase):
    def test_uniform_scale_and_resolution(self):
        m=PriorPrompt(dim=8, backprojection="patch_mean_compensated")
        values=torch.randn(3,8)
        for patches in (20,4500):
            a=torch.full((3,patches),1/patches)
            torch.testing.assert_close(m.backproject(a,values),values.mean(0).expand(patches,-1))

    def test_single_object_keeps_spatial_response(self):
        m=PriorPrompt(dim=8,backprojection="patch_mean_compensated")
        a=torch.tensor([[.1,.2,.3,.4]]);p=torch.ones(1,8)
        torch.testing.assert_close(m.backproject(a,p),4*a.T@p)
        self.assertFalse(torch.equal(m.backproject(a,p),m.backproject(a.flip(1),p)))

    def test_nonzero_initialization_and_empty_identity(self):
        torch.manual_seed(0)
        m=PriorPrompt(dim=8,patch=4,backprojection="patch_mean_compensated",projection_init_gain=.01)
        x=torch.randn(1,3,16,20)
        self.assertTrue(torch.equal(x,m(x,torch.empty(0,4),torch.empty(0,dtype=torch.long))))
        y=m(x,torch.tensor([[1.,2.,3.,4.]]),torch.tensor([1]))
        self.assertFalse(torch.equal(x,y))
        y.square().mean().backward()
        self.assertGreater(m.key.weight.grad.abs().max().item(),0)

if __name__=='__main__':unittest.main()
