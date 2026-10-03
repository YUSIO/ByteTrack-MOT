import unittest
from types import SimpleNamespace
import numpy as np
import torch
from homa.model import MPANet, association_loss, tracklet_similarity
from homa.association import HOMAAssociation
from yolox.tracker.byte_tracker import STrack, BYTETracker
from yolox.tracker.basetrack import BaseTrack


class HomaTests(unittest.TestCase):
    def test_frame_softmax_and_online_owner_sum(self):
        logits=torch.zeros(1,3)
        p=tracklet_similarity(logits,torch.tensor([1,1,2]),torch.tensor([0,1,0]),2)
        torch.testing.assert_close(p,torch.tensor([[1.5,.5]]))

    def test_association_masks_missing_and_has_gradients(self):
        parts=torch.randn(3,2,4,requires_grad=True); fused=torch.randn(2,4,requires_grad=True)
        loss,n=association_loss(parts,fused,torch.tensor([1,8]),torch.tensor([1,2,1,3]),torch.tensor([1,1,2,2]))
        self.assertEqual(n,2); loss.backward(); self.assertGreater(float(parts.grad.abs().sum()),0)
        self.assertEqual(float(parts.grad[:,1].abs().sum()),0)

    def test_motion_sum_and_window_expiry(self):
        t=SimpleNamespace(track_id=1,mean=np.array([10.,10.]),covariance=np.eye(2))
        d=STrack(np.array([9.,9.,2.,2.]),.9)
        a=HOMAAssociation(None,'m2da',motion_window=3,kappa=.1)
        a.frame=1; np.testing.assert_allclose(a.motion_similarity([t],[d]),[[1.]])
        t.mean=np.array([11.,10.]); a.frame=2
        np.testing.assert_allclose(a.motion_similarity([t],[d]),[[.1/1.1]])
        a.frame=3
        np.testing.assert_allclose(a.motion_similarity([t],[d]),[[.1/2.1]])

    def test_history_expires_by_frame_not_observation_count(self):
        a=HOMAAssociation(None,'homa',window=8)
        a.history.append((1,[(1,torch.zeros(8192))])); a.set_current(torch.zeros(0,8192),9)
        self.assertFalse(a.history)

    def test_low_score_update_does_not_refresh_visual_history(self):
        args=SimpleNamespace(track_thresh=.6,track_buffer=30,match_thresh=.9,mot20=False)
        BaseTrack._count=0; t=BYTETracker(args)
        a=HOMAAssociation(None,'m2da'); t.homa=a
        a.set_current(torch.ones(1,8192),1)
        t.update(np.array([[0.,0.,20.,20.,.9]]),(100,100),(100,100))
        self.assertEqual(len(a.history),1)
        a.set_current(torch.empty(0,8192),2)
        t.update(np.array([[0.,0.,20.,20.,.4]]),(100,100),(100,100))
        self.assertEqual(len(a.history),1)
        self.assertEqual(t.tracked_stracks[0].homa_feature_frame,1)

    def test_amp_overflow_skips_and_recovers(self):
        from homa.train import finish_optimizer_step
        model=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad(): model.weight.fill_(1.)
        opt=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9)
        scaler=torch.amp.GradScaler('cpu',init_scale=8.)
        scaler.scale(model(torch.tensor([[float('inf')]])).sum()).backward()
        self.assertTrue(finish_optimizer_step(model,opt,scaler,1,1))
        self.assertEqual(float(model.weight.detach()),1.)
        self.assertFalse(opt.state)
        self.assertEqual(scaler.get_scale(),4.)
        scaler.scale(model(torch.tensor([[2.]])).sum()).backward()
        self.assertFalse(finish_optimizer_step(model,opt,scaler,1,1))
        self.assertAlmostEqual(float(model.weight.detach()),.8,places=6)

    def test_model_finite_backward(self):
        torch.set_num_threads(2); torch.manual_seed(42)
        m=MPANet(); m.train()
        parts,fused=m(torch.randn(4,3,128,128),2)
        self.assertEqual(tuple(parts.shape),(3,2,2))
        loss,n=association_loss(parts,fused,torch.tensor([1,2]),torch.tensor([1,2]),torch.tensor([1,1])); loss.backward()
        for group in (m.backbone,m.patch,m.pose,m.fuse):
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in group.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))


if __name__=='__main__':unittest.main()
