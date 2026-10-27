import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from feedback.trades.history_counterfactual import zero_history_support,cache_detections,HistoryReplay
from feedback.trades.model import TraDeSAdapter,FrozenTraDeS
from feedback.trades.preflight import tensor_hashes


class HistoryCounterfactualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3);torch.set_num_threads(2)

    def test_zero_support_keeps_alignment_and_state_then_restores(self):
        model=TraDeSAdapter(8,(4,6)).eval()
        with torch.no_grad():model.output.weight.normal_(0,.1)
        current=torch.randn(1,8,4,6);previous=torch.randn_like(current);heat=torch.ones(1,1,4,6)
        before=tensor_hashes(model)
        observed=[]
        def record(module,args):observed.append(tuple(x.clone() for x in args))
        with torch.no_grad():
            base=model(current,previous,heat)
            hook=model.core.dcn1_1.register_forward_pre_hook(record)
            model(current,previous,heat);hook.remove()
            with zero_history_support(model.core,True):
                hook=model.core.dcn1_1.register_forward_pre_hook(record)
                zero=model(current,previous,heat);hook.remove()
            restored=model(current,previous,heat)
        self.assertEqual(float(observed[1][0].abs().sum()),0.)
        for a,b in zip(observed[0][1:],observed[1][1:]):self.assertTrue(torch.equal(a,b))
        for axis in ['h','w','offset']:self.assertTrue(torch.equal(base[1][axis],zero[1][axis]))
        self.assertTrue(torch.equal(base[0],restored[0]))
        self.assertFalse(torch.equal(base[0],zero[0]))
        self.assertEqual(tensor_hashes(model),before)

    def test_hook_removed_on_exception_and_empty_history_still_bypasses(self):
        model=TraDeSAdapter(8,(4,6)).eval();current=torch.randn(1,8,4,6)
        with self.assertRaises(RuntimeError):
            with zero_history_support(model.core,True):raise RuntimeError('test')
        self.assertEqual(len(model.core.dcn1_1._forward_pre_hooks),0)
        with zero_history_support(model.core,True):
            out,cva,_=model(current,current,torch.zeros(1,1,4,6))
        self.assertIsNone(cva);self.assertTrue(torch.equal(out,current))

    def test_replay_has_confirmation_delay_and_no_intervention_input(self):
        replay=HistoryReplay();shape=(100,100)
        replay.advance(np.empty((0,5)),shape,1)
        replay.advance(np.array([[10,10,20,20,.9]]),shape,2)
        self.assertEqual(len(replay.boxes),0)
        replay.advance(np.array([[10,10,20,20,.9]]),shape,3)
        np.testing.assert_array_equal(replay.boxes,[[10,10,20,20]])
        with self.assertRaises(ValueError):replay.advance(np.empty((0,5)),shape,5)

    def test_cache_keeps_empty_frames_and_xywh_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'det.txt';path.write_text('2,-1,1.2345,2.3456,3.4567,4.5678,0.765432,-1,-1,-1\n')
            frames=cache_detections(path,3)
        self.assertEqual(frames[1].shape,(0,5));self.assertEqual(frames[3].shape,(0,5))
        np.testing.assert_allclose(frames[2],[[1.2345,2.3456,4.6912,6.9134,.765432]],atol=1e-12)

    def test_shared_features_and_loss_dispatch_preserve_predictions(self):
        model=FrozenTraDeS(architecture=dict(depth=.33,width=.125),input_size=(64,96)).eval()
        image=torch.randn(1,3,64,96);previous=torch.randn_like(image);heat=torch.ones(1,1,8,12)
        head=model.detector.head;head.decode_in_inference=False
        with torch.no_grad():
            model.adapter.output.weight.normal_(0,.01)
            before=tensor_hashes(model)
            expected=model(image,previous_images=previous,heat=heat)['output']
            features=model.detector.backbone(image);old=model.detector.backbone(previous)[0]
            refined,_,_=model.adapter(features[0],old,heat)
            level=(refined,*features[1:]);prediction=head(level)
            self.assertTrue(torch.equal(prediction,expected))
            try:
                head.training=True
                losses=head(level,torch.tensor([[[0.,32.,32.,16.,16.]]]),image)
            finally:head.training=False
            self.assertTrue(torch.isfinite(losses[0]))
            self.assertTrue(torch.equal(prediction,head(level)))
            self.assertEqual(before,tensor_hashes(model))


if __name__=='__main__':unittest.main()
