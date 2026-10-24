import unittest
import torch
from feedback.prompt import PriorPrompt

class CosineContentTests(unittest.TestCase):
    def test_bounded_and_scale_invariant_content(self):
        torch.manual_seed(3)
        m=PriorPrompt(position_encoding=True,content_score='cosine')
        q,k=torch.randn(7,64),torch.randn(24,64)
        score=m.content_logits(q,k)
        self.assertLessEqual(float(score.abs().max()),1)
        torch.testing.assert_close(score,m.content_logits(q*100,k*.01))
        self.assertTrue(torch.isfinite(m.content_logits(torch.zeros_like(q),torch.zeros_like(k))).all())

    def test_forward_uses_same_logits_and_backward(self):
        torch.manual_seed(4)
        m=PriorPrompt(position_encoding=True,content_score='cosine')
        torch.nn.init.normal_(m.to_patch.weight,std=.1)
        x=torch.randn(1,3,64,96,requires_grad=True)
        boxes=torch.tensor([[0.,0.,16.,16.],[32.,16.,16.,16.]])
        ids=torch.tensor([1,2])
        q=m.encode_query(boxes,ids,64,96)
        patches=m.patch_embed(x).flatten(2).transpose(1,2)[0]
        expected=m.content_logits(q,m.key(patches))+q@m.patch_positions(64,96,patches).T/8
        torch.testing.assert_close(m.attention(q,patches,64,96),expected.softmax(-1))
        y=m(x,boxes,ids);y.square().mean().backward()
        for name in ['key.weight','patch_embed.weight','coordinate.0.weight']:
            p=dict(m.named_parameters())[name]
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs().sum()),0)

    def test_zero_projection_and_empty_remain_identity(self):
        m=PriorPrompt(position_encoding=True,content_score='cosine')
        x=torch.randn(1,3,32,48)
        torch.testing.assert_close(m(x,torch.tensor([[0.,0.,16.,16.]]),torch.tensor([1])),x)
        torch.testing.assert_close(m(x,torch.empty(0,4),torch.empty(0,dtype=torch.long)),x)
