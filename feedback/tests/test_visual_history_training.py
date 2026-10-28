import tempfile
import unittest
from pathlib import Path
import torch
from feedback.trades.model import TraDeSAdapter, FrozenTraDeS
from feedback.trades.fusion_terms import frozen_fusion_terms
from feedback.trades.history_counterfactual import zero_history_support
from feedback.trades.checkpoint import save_adapter, load_adapter
from feedback.trades.preflight import tensor_hashes


class VisualHistoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(11)

    def test_visual_formula_matches_fixed_native_attention_intervention(self):
        model=TraDeSAdapter(8,(4,6)).eval()
        with torch.no_grad():model.output.weight.normal_(0,.1)
        cur=torch.randn(1,8,4,6);prev=torch.randn_like(cur);heat=torch.rand(1,1,4,6)
        expected=frozen_fusion_terms(model,cur,prev,heat)[0]['visual_only']
        model.fusion_mode='visual_only'
        with torch.no_grad():actual=model(cur,prev,heat)[0]
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_zero_visual_support_returns_off_with_nonzero_bias(self):
        model=TraDeSAdapter(8,(4,6));model.fusion_mode='visual_only'
        with torch.no_grad():model.output.weight.normal_(0,.1);model.core.dcn1_1.bias.fill_(.2)
        cur=torch.randn(2,8,4,6);prev=torch.randn_like(cur);heat=torch.ones(2,1,4,6)
        for training in [True,False]:
            model.train(training)
            with zero_history_support(model.core,True):out=model(cur,prev,heat)[0]
            self.assertTrue(torch.equal(out,cur))

    def test_detection_backward_updates_branch_and_freezes_detector(self):
        model=FrozenTraDeS(architecture=dict(depth=.33,width=.125),input_size=(64,96))
        model.adapter.fusion_mode='visual_only'
        with torch.no_grad():model.adapter.output.weight.normal_(0,.05)
        frozen=tensor_hashes(model.detector)
        image=torch.randn(1,3,64,96);old=torch.randn_like(image)
        out=model(image,previous_images=old,heat=torch.ones(1,1,8,12),targets=torch.tensor([[[0.,32.,32.,16.,16.]]]))
        loss=out['output']['total_loss'];self.assertTrue(torch.isfinite(loss));loss.backward()
        for module in [model.adapter.output,model.adapter.core.dcn1_1,model.adapter.core.attention_prev]:
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None),0)
        self.assertTrue(all(p.grad is None for p in model.detector.parameters()))
        torch.optim.SGD(model.optimizer_parameters(),lr=1e-4).step()
        self.assertEqual(tensor_hashes(model.detector),frozen)

    def test_checkpoint_preserves_fusion_and_legacy_defaults_full(self):
        model=FrozenTraDeS(architecture=dict(depth=.33,width=.125),input_size=(64,96))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base';base.write_bytes(b'fixed-base-hash-fixture')
            model.adapter.fusion_mode='visual_only';save_adapter(model,root/'new.pt',base,dict(epoch=1))
            model.adapter.fusion_mode='full';load_adapter(model,root/'new.pt',base)
            self.assertEqual(model.adapter.fusion_mode,'visual_only')
            old=torch.load(root/'new.pt',weights_only=True);old.pop('fusion_mode');torch.save(old,root/'old.pt')
            load_adapter(model,root/'old.pt',base);self.assertEqual(model.adapter.fusion_mode,'full')


if __name__=='__main__':unittest.main()
