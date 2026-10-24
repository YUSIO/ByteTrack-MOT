import unittest
from types import SimpleNamespace
import numpy as np
import torch
from feedback.model import build
from feedback.online import OnlineFeedback
from feedback.state import observe_detector_boxes

TRACKER={'track_thresh':.6,'track_buffer':30,'match_thresh':.9,'mot20':False}


class IntegrationTests(unittest.TestCase):
    def test_frozen_detector_preserves_input_gradient_and_weights(self):
        from feedback.train import configure_detector_training, backward_loss
        from feedback.learning_gate import state_hash
        m=build({'mode':'pp','architecture':{'depth':.33,'width':.25},
                 'prompt':{'dim':8,'patch':4,'bins':16,'max_ids':16,'projection_init_gain':.01}})
        configure_detector_training(m,True)
        before=state_hash(m.detector)
        opt=torch.optim.SGD([p for p in m.parameters() if p.requires_grad],lr=.001)
        x=torch.randn(2,3,64,96)
        labels=torch.tensor([[[0.,30.,25.,8.,6.]],[[0.,35.,25.,8.,6.]]])
        contexts=[{'boxes':torch.tensor([[25.,20.,8.,6.]]),'ids':torch.tensor([1])}]*2
        m(x,labels,contexts)['total_loss'].backward()
        self.assertGreater(m.feedback.key.weight.grad.abs().sum().item(),0)
        self.assertGreater(m.feedback.to_patch.weight.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None for p in m.detector.parameters()))
        opt.step();self.assertEqual(before,state_hash(m.detector))
        opt.zero_grad(set_to_none=True)
        feedback_before=state_hash(m.feedback)
        loss=m(x,labels,[None,None])['total_loss']
        self.assertFalse(backward_loss(loss,True))
        opt.step()
        self.assertEqual(feedback_before,state_hash(m.feedback))
        self.assertEqual(before,state_hash(m.detector))

    def test_feedback_off_equals_original_yolox(self):
        torch.manual_seed(1);torch.set_num_threads(2)
        m=build({'mode':'none','architecture':{'depth':.33,'width':.25}}).eval()
        x=torch.randn(1,3,64,96)
        with torch.no_grad():
            expected=m.detector(x)
            actual=m(x)
        self.assertTrue(torch.equal(expected,actual))

    def test_prompt_training_reaches_backbone_and_head(self):
        m=build({'mode':'pp','architecture':{'depth':.33,'width':.25},
                 'prompt':{'dim':8,'patch':4,'bins':16,'max_ids':16}}).train()
        for layer in m.modules():
            if isinstance(layer,torch.nn.BatchNorm2d):layer.eval()
        torch.nn.init.normal_(m.feedback.to_patch.weight,std=.01)
        x=torch.randn(2,3,64,96)
        labels=torch.tensor([[[0.,30.,25.,8.,6.]],[[0.,35.,25.,8.,6.]]])
        contexts=[{'boxes':torch.tensor([[25.,20.,8.,6.]]),'ids':torch.tensor([1])}]*2
        out=m(x,labels,contexts);out['total_loss'].backward()
        self.assertTrue(torch.isfinite(out['total_loss']))
        for parameter in [m.feedback.key.weight,m.detector.backbone.backbone.stem.conv.conv.weight,
                          m.detector.head.reg_preds[0].weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(),0)

    def test_observation_hook_does_not_change_tracker(self):
        from yolox.tracker.byte_tracker import BYTETracker
        from yolox.tracker.basetrack import BaseTrack
        frames=[np.array([[t,5,t+10,15,.9]],float) if t not in (4,5) else np.empty((0,5))
                for t in range(1,10)]
        def run():
            BaseTrack._count=0;t=BYTETracker(SimpleNamespace(**TRACKER),30);out=[]
            for f in frames:
                rows=t.update(f.copy(),(100,100),(100,100))
                out.append([(s.track_id,s.tlwh.tolist(),s.mean.tolist()) for s in rows])
            return out
        before=run();observe_detector_boxes();after=run();self.assertEqual(before,after)


if __name__=='__main__':unittest.main()
