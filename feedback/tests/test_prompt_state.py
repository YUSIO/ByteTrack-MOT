import unittest
import numpy as np
import torch
from feedback.prompt import PriorPrompt
from feedback.state import History
from feedback.model import FeedbackDetector


class PromptTests(unittest.TestCase):
    def test_empty_and_initial_are_exact_identity(self):
        m=PriorPrompt(dim=8,patch=4,bins=16,max_ids=8)
        x=torch.randn(1,3,17,19)
        self.assertTrue(torch.equal(x,m(x,torch.empty(0,4),torch.empty(0,dtype=torch.long))))
        self.assertTrue(torch.equal(x,m(x,torch.tensor([[3.,4.,2.,2.]]),torch.tensor([1]))))

    def test_nonzero_attention_has_gradients_and_box_dependence(self):
        torch.manual_seed(9)
        m=PriorPrompt(dim=8,patch=4,bins=16,max_ids=8)
        torch.nn.init.normal_(m.to_patch.weight,std=.1)
        x=torch.randn(1,3,16,20,requires_grad=True)
        boxes=torch.tensor([[1.,2.,3.,4.],[8.,7.,2.,3.]])
        y=m(x,boxes,torch.tensor([1,2]))
        changed=m(x,boxes+1,torch.tensor([1,2]))
        self.assertFalse(torch.allclose(y,changed))
        y.square().mean().backward()
        for name,p in m.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
        self.assertGreater(m.key.weight.grad.abs().sum().item(),0)
        perm=m(x.detach(),boxes.flip(0),torch.tensor([2,1]))
        torch.testing.assert_close(y.detach(),perm)

    def test_no_id_wrap(self):
        m=PriorPrompt(dim=8,patch=4,bins=16,max_ids=8)
        with self.assertRaises(ValueError):
            m(torch.zeros(1,3,16,16),torch.tensor([[1.,1.,2.,2.]]),torch.tensor([8]))


class HistoryTests(unittest.TestCase):
    def test_only_contiguous_observations(self):
        h=History(3)
        for t in range(1,4):h.append(t,{1:[t,0,2,2]})
        p,ids=h.reliable();self.assertEqual(ids,[1]);np.testing.assert_equal(p[0,:,0],[2,3,4])
        h.append(4,{});self.assertEqual(len(h.reliable()[1]),0)
        for t in range(5,7):h.append(t,{1:[t,0,2,2]})
        self.assertEqual(len(h.reliable()[1]),0)
        h.append(7,{1:[7,0,2,2]});self.assertEqual(h.reliable()[1],[1])
        with self.assertRaises(ValueError):h.append(9,{})
        h.reset();self.assertEqual(h.frame,0);self.assertEqual(h.reliable()[1],[])

    def test_copy_not_mutable_state(self):
        h=History(2); box=np.array([1,2,3,4.]);h.append(1,{1:box});box[0]=100
        self.assertEqual(h.frames[0][1][0],1)


if __name__=='__main__':unittest.main()
