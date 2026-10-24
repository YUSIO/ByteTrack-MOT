import unittest
import torch
from torch import nn
from feedback.online import OnlineFeedback
from feedback.model import build

CONFIG={'tracker':{'track_thresh':.6,'track_buffer':30,'match_thresh':.9,'mot20':False},'window':8}


class FakeDetector(nn.Module):
    def __init__(self):
        super().__init__();self.mode='eec';self.contexts=[]
        self.detector=nn.Module();self.detector.head=nn.Module()
        self.detector.head.hw=[(1,1)];self.detector.head.strides=[8]

    def forward(self,image,contexts,return_raw):
        self.contexts.append(contexts[0])
        # Stable observation far from zero boundary, high confidence.
        raw=image.new_tensor([[[3.,3.,0.,0.,.99,.99]]])
        feature=image.new_full((1,2,4,4),len(self.contexts))
        return raw,feature


class FakePSMP(nn.Module):
    def __init__(self):super().__init__();self.inputs=[]
    def forward(self,history):
        self.inputs.append(history.clone())
        return {'prior':{'velocity':history.new_zeros((1,2)), 'mean':history.new_zeros((1,2)),
                         'covariance':torch.eye(2)[None].to(history)}}


class CausalityTests(unittest.TestCase):
    def test_eight_frame_warmup_raw_cache_and_reset(self):
        model=FakeDetector();predictor=FakePSMP();run=OnlineFeedback(model,CONFIG,predictor)
        image=torch.zeros(1,3,32,32)
        results=[run.step(image,(32,32),1) for _ in range(10)]
        self.assertEqual([r['feedback_used'] for r in results],[False]*8+[True]*2)
        self.assertEqual(len(predictor.inputs),2)
        self.assertEqual(predictor.inputs[0].shape,(1,1,8,2))
        self.assertTrue(torch.equal(model.contexts[8]['previous'],torch.full((1,2,4,4),8.)))
        self.assertTrue(torch.equal(model.contexts[9]['previous'],torch.full((1,2,4,4),9.)))
        self.assertEqual(results[8]['history_last_frame_used'],8)
        with self.assertRaises(ValueError):run.step(image,(64,64),.5)
        run.reset();r=run.step(image,(32,32),1)
        self.assertFalse(r['feedback_used']);self.assertEqual(r['frame'],1)

    def test_eec_detector_joint_gradient_and_no_feedback_bypass(self):
        torch.set_num_threads(2)
        m=build({'mode':'eec','architecture':{'depth':.33,'width':.25},
                 'eec':{'channels':64,'window':3}}).train()
        for layer in m.modules():
            if isinstance(layer,nn.BatchNorm2d):layer.eval()
        nn.init.normal_(m.feedback.fusion.weight,std=.01)
        image=torch.randn(2,3,64,96);old=torch.randn_like(image,requires_grad=True)
        labels=torch.tensor([[[0.,30.,25.,8.,6.]],[[0.,35.,25.,8.,6.]]])
        prior={'velocity':torch.zeros(1,2),'mean':torch.zeros(1,2),'covariance':torch.eye(2)[None]*64}
        loss=m(image,labels,[{'prior':prior},None],old)['total_loss'];loss.backward()
        for p in [old,m.feedback.alpha,m.feedback.fusion.weight,m.detector.head.reg_preds[0].weight]:
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(),0)
        m.eval()
        with torch.no_grad():
            expected=m.detector(image);actual=m(image,contexts=[None,None])
        self.assertTrue(torch.equal(expected,actual))


if __name__=='__main__':unittest.main()
