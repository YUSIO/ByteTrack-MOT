"""Numerical regression and gradient check; no data or GPU required."""
import json
import torch
from yolox.models.losses import IOUloss
p = torch.tensor([[0.,0.,2.,2.],[.4,0.,2.,2.],[0.,0.,1.,1.],[5.,5.,1.,1.]], dtype=torch.float64, requires_grad=True)
t = torch.tensor([[0.,0.,2.,2.]]*4, dtype=torch.float64)
u = torch.tensor([1., 2./3., .25, 0.], dtype=torch.float64)
a = IOUloss()(p,t)
b = IOUloss(iou_alpha=3)(p,t)
torch.testing.assert_close(a, 1-u**2)
torch.testing.assert_close(b, 1-u**3)
ga = torch.autograd.grad(a.sum(),p,retain_graph=True)[0]
gb = torch.autograd.grad(b.sum(),p)[0]
torch.testing.assert_close(gb, ga * (1.5*u[:,None]))
assert torch.isfinite(gb).all()
print(json.dumps({'status':'passed','alpha2':a.tolist(),'alpha3':b.tolist(),'gradient_ratio':'1.5*IoU','cases':['perfect','partial','contained','disjoint']}))
