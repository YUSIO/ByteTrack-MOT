import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from feedback.trades.model import FrozenTraDeS,TraDeSAdapter
from feedback.trades.data import correspondence_targets,cva_loss,render_heat
from feedback.trades.vendor.ops import DCN_TraDeS
from feedback.trades.online import TraDeSOnline


def official_loss_class():
    root=Path(__file__).parents[1]/'trades/vendor'
    ns=dict(torch=torch,nn=nn,F=F)
    for filename,names in [('utils_source.py',{'_gather_feat','_tranpose_and_gather_feat_1d'}),
                           ('loss_source.py',{'_only_neg_loss','CostVolumeLoss1D'})]:
        tree=ast.parse((root/filename).read_text())
        selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        exec(compile(ast.Module(body=selected,type_ignores=[]),filename,'exec'),ns)
    return ns['CostVolumeLoss1D']()


class TraDeSTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0);torch.set_num_threads(2)

    def test_dcn_xy_order_and_backward_sampling(self):
        m=DCN_TraDeS(1,1,3,1,1);x=torch.arange(20.).reshape(1,1,4,5)
        offset=torch.zeros(1,18,4,5);mask=torch.ones(1,9,4,5)
        torch.testing.assert_close(m(x,offset,mask),F.conv2d(x,m.weight,m.bias,padding=1))
        with torch.no_grad():m.weight.zero_();m.weight[0,0,1,1]=1;m.bias.zero_()
        offset[:,9]=1
        expected=torch.zeros_like(x);expected[:,:,:,:-1]=x[:,:,:,1:]
        torch.testing.assert_close(m(x,offset,mask),expected)

    def test_cva_known_correspondence_direction(self):
        m=TraDeSAdapter(64,(4,6)).core
        m.embedconv=nn.Identity();m.embed_dim=6
        coarse=torch.eye(6).T.reshape(1,6,2,3)*10
        prev=coarse.roll(1,3).repeat_interleave(2,2).repeat_interleave(2,3)
        _,_,offset=m.CVA(coarse.flatten(2).transpose(1,2),prev,1,2,3)
        torch.testing.assert_close(offset[0,0],torch.tensor([2.,2.,-4.]).repeat_interleave(2).repeat(4,1))
        torch.testing.assert_close(offset[0,1],torch.zeros(4,6))

    def test_loss_matches_official_and_gradients(self):
        now={1:np.array([16,16,8,8]),2:np.array([48,32,12,12])}
        old={1:np.array([32,16,8,8]),2:np.array([32,32,12,12])}
        target,stats=correspondence_targets(now,old,1,(64,96))
        self.assertEqual(stats['used_pairs'],2)
        h=torch.randn(1,24,4,requires_grad=True);w=torch.randn(1,24,6,requires_grad=True)
        got=cva_loss(dict(h=h,w=w,active=torch.tensor([True])),[target])
        original=official_loss_class()
        expected=sum(original(out,target[axis+'_hm_down'][None],target[axis+'_ind_down'][None],
                              target['mask_down'][None],target['cat_down'][None])
                     for axis,out in [('h',h),('w',w)])
        torch.testing.assert_close(got,expected)
        gradients=torch.autograd.grad(got,[h,w],retain_graph=True)
        original_gradients=torch.autograd.grad(expected,[h,w])
        for a,b in zip(gradients,original_gradients):torch.testing.assert_close(a,b)

    def test_collision_exclusion_is_order_independent(self):
        now={1:[1,1,4,4],2:[3,3,4,4],3:[32,32,4,4]}
        old={1:[17,1,4,4],2:[33,3,4,4],3:[48,32,4,4]}
        a,stats=correspondence_targets(now,old,1,(64,96))
        b,_=correspondence_targets(dict(reversed(list(now.items()))),old,1,(64,96))
        self.assertEqual(stats,dict(common_valid_pairs=3,used_pairs=1,collision_excluded_pairs=2))
        for key in a:torch.testing.assert_close(a[key],b[key])

    def test_identity_bypass_after_nonzero_update(self):
        a=TraDeSAdapter(8,(8,12)).eval()
        x=torch.randn(2,8,8,12);old=torch.randn_like(x);heat=torch.ones(2,1,8,12)
        self.assertTrue(torch.equal(a(x,old,heat)[0],x))
        with torch.no_grad():a.output.weight.normal_(0,.05)
        self.assertIs(a(x,old,heat,False)[0],x)
        self.assertIs(a(x,None,heat)[0],x)
        self.assertIs(a(x,old,heat*0)[0],x)
        heat[0]=0
        out,cva,_=a(x,old,heat)
        self.assertTrue(torch.equal(out[0],x[0]));self.assertFalse(torch.equal(out[1],x[1]))
        self.assertEqual(cva['active'].tolist(),[False,True])

    def test_detector_frozen_bn_and_detection_gradient(self):
        m=FrozenTraDeS(architecture={'depth':.33,'width':.25},input_size=(64,96)).train()
        before={k:v.clone() for k,v in m.detector.state_dict().items()}
        x=torch.randn(1,3,64,96);old=torch.randn_like(x)
        heat=torch.ones(1,1,8,12);gt=torch.tensor([[[0.,30,30,12,12]]])
        opt=torch.optim.SGD(m.optimizer_parameters(),lr=1e-3)
        with torch.no_grad():m.adapter.output.weight.normal_(0,.01)
        result=m(x,previous_images=old,heat=heat,targets=gt)
        result['output']['total_loss'].backward()
        for p in m.detector.parameters():self.assertIsNone(p.grad)
        for name in ['input.weight','output.weight','core.dcn1_1.weight','core.conv_offset_w.weight']:
            g=dict(m.adapter.named_parameters())[name].grad
            self.assertTrue(torch.isfinite(g).all(),name);self.assertGreater(g.abs().sum(),0,name)
        opt.step();m.eval()
        for k,v in m.detector.state_dict().items():self.assertTrue(torch.equal(before[k],v),k)
        with torch.no_grad():
            base=m.detector(x)
            for history in [heat,heat*0]:
                self.assertTrue(torch.equal(base,m(x,previous_images=old,heat=history,enabled=False)['output']))
            self.assertTrue(torch.equal(base,m(x,previous_images=old,heat=heat*0)['output']))

    def test_heat_and_no_future_online_state(self):
        self.assertEqual(render_heat([], (64,96)).sum(),0)
        heat=render_heat([[16,16,24,24]],(64,96))
        self.assertGreater(heat.sum(),0);self.assertLessEqual(heat.max(),1)
        m=FrozenTraDeS(architecture={'depth':.33,'width':.25},input_size=(64,96)).eval()
        runner=TraDeSOnline(m);runner.reset('synthetic')
        x=torch.randn(1,3,64,96)
        r=runner.step(x,'synthetic',1,(64,96),1.)
        self.assertFalse(r['feedback_used']);self.assertEqual(r['history_frame'],0)
        with self.assertRaises(ValueError):runner.step(x,'synthetic',1,(64,96),1.)
        with self.assertRaises(ValueError):runner.step(x,'other',2,(64,96),1.)
        runner.reset('other');self.assertIsNone(runner.previous)


if __name__=='__main__':unittest.main()
