import unittest
import torch
from feedback.model import build
from feedback.train import configure_detector_training
from feedback.fixed_matching import fixed_matching_loss, matching_context
from feedback.learning_gate import state_hash


class FixedMatchingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(11)
        self.model = build({'mode':'pp','architecture':{'depth':.33,'width':.25},
            'prompt':{'dim':8,'patch':4,'bins':16,'max_ids':16,'projection_init_gain':.01}})
        configure_detector_training(self.model, True)
        self.x = torch.randn(2,3,64,96)
        self.y = torch.tensor([[[0.,30.,25.,8.,6.]],[[0.,35.,25.,8.,6.]]])

    def test_off_equivalence_empty_gt_and_restoration(self):
        for y in (self.y, torch.zeros_like(self.y), torch.cat((self.y[:1],torch.zeros_like(self.y[:1])))):
            with torch.no_grad():
                expected=self.model(self.x,y,[None,None])
                actual=fixed_matching_loss(self.model,self.x,y,[None,None])
            for key in expected:
                torch.testing.assert_close(torch.as_tensor(actual[key]),torch.as_tensor(expected[key]),rtol=1e-6,atol=1e-6)
            self.assertNotIn('get_assignments',self.model.detector.head.__dict__)
        with self.assertRaisesRegex(ValueError,'sentinel'):
            with matching_context(self.model.detector.head,{},capture=True):
                raise ValueError('sentinel')
        self.assertNotIn('get_assignments',self.model.detector.head.__dict__)

    def test_gradient_support_and_live_iou_targets(self):
        head=self.model.detector.head;matches={}
        with torch.no_grad(),matching_context(head,matches,capture=True):
            self.model(self.x,self.y,[None,None])
        item=matches[0];old=item['result'];pred=torch.zeros(item['anchors'],4)
        pred[old[1]]=item['gt'][old[3]]
        args=(0,len(item['gt']),item['anchors'],item['gt'],item['classes'],pred)
        with matching_context(head,matches):
            got=head.get_assignments(*args)
            self.assertTrue(torch.equal(got[1],old[1]))
            self.assertTrue(torch.equal(got[3],old[3]))
            self.assertEqual(got[4],old[4])
            torch.testing.assert_close(got[2],torch.ones_like(got[2]))
            self.assertFalse(got[2].requires_grad)
            with self.assertRaises(ValueError):
                head.get_assignments(*args[:3],args[3]+1,*args[4:])
        before=state_hash(self.model.detector)
        ctx=[{'boxes':torch.tensor([[25.,20.,8.,6.]]),'ids':torch.tensor([1])}]*2
        loss=fixed_matching_loss(self.model,self.x,self.y,ctx)['total_loss'];loss.backward()
        self.assertGreater(self.model.feedback.to_patch.weight.grad.abs().sum().item(),0)
        self.assertGreater(self.model.feedback.key.weight.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None for p in self.model.detector.parameters()))
        self.assertEqual(before,state_hash(self.model.detector))


if __name__=='__main__':unittest.main()
