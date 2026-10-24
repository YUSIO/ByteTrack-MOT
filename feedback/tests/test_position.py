import unittest
import torch
from feedback.prompt import PriorPrompt

class PositionTests(unittest.TestCase):
    def test_position_key_matches_grid_order_and_has_no_parameters(self):
        m=PriorPrompt(position_encoding=True)
        pe=m.patch_positions(32,48,torch.zeros(6,64))
        self.assertEqual(pe.shape,(6,64))
        torch.testing.assert_close(pe[0,32:],pe[1,32:])
        torch.testing.assert_close(pe[0,:32],pe[3,:32])
        self.assertFalse(torch.equal(pe[0],pe[1]))
        self.assertEqual(sum(p.numel() for p in m.parameters()),sum(p.numel() for p in PriorPrompt().parameters()))

    def test_position_can_select_same_content_by_aligned_query(self):
        m=PriorPrompt(position_encoding=True).double()
        patches=torch.zeros(24,64,dtype=torch.float64)
        q=m.patch_positions(64,96,patches)
        a=m.attention(q,patches,64,96)
        self.assertTrue(torch.equal(a.argmax(-1),torch.arange(24)))
        m.position_encoding=False
        a=m.attention(q,patches,64,96)
        torch.testing.assert_close(a,torch.full_like(a,1/24))

    def test_zero_residual_and_empty_prior_unchanged(self):
        m=PriorPrompt(position_encoding=True)
        x=torch.randn(1,3,32,48)
        torch.testing.assert_close(m(x,torch.tensor([[0.,0.,16.,16.]]),torch.tensor([1])),x)
        torch.testing.assert_close(m(x,torch.empty(0,4),torch.empty(0,dtype=torch.long)),x)
