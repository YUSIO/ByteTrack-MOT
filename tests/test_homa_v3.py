import unittest
import torch
from homa.model import ResidualNormPoseBlock
from homa.train import uniform_window_loss


class SignalTests(unittest.TestCase):
    def test_query_survives_zero_attention_and_small_scale(self):
        torch.manual_seed(42)
        m = ResidualNormPoseBlock(16, heads=4, hidden=16)
        for a in (m.sa, m.ca):
            for p in a.parameters():
                torch.nn.init.zeros_(p)
        q = (torch.randn(3,16)*.04).requires_grad_()
        h = torch.randn(4,16)*.04
        with torch.autocast('cpu',dtype=torch.bfloat16):
            logits = m(q,h)
        self.assertEqual(logits.dtype,torch.float32)
        self.assertGreater(float((logits[0]-logits[1]).abs().max()),1e-4)
        logits.square().sum().backward()
        self.assertGreater(float(q.grad.norm()),1e-4)

    def test_uniform_baseline_unequal_identity_presence(self):
        sample={'all_ids':torch.tensor([1,2,1]),'all_frames':torch.tensor([1,1,2])}
        self.assertAlmostEqual(uniform_window_loss(sample),2*float(torch.log(torch.tensor(2.))),places=6)


if __name__=='__main__': unittest.main()
