"""CPU-only analytic tests and base-source invariance, no dataset access."""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = 'ee7b677cc25c8070f45ce89bf63b532089dc7655'

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

NWD = load(ROOT/'yolox/models/nwd_loss.py', 'isolated_nwd').NWDloss
old = load(ROOT/'yolox/models/losses.py', 'isolated_iou').IOUloss
cases = [
    ('identical', [10,10,2,2], [10,10,2,2]),
    ('partial', [10,10,4,4], [11,12,4,4]),
    ('contained', [10,10,8,8], [10,10,2,2]),
    ('disjoint', [10,10,2,2], [30,40,2,2]),
    ('tiny', [1,1,1,1], [2,1,1,1]),
    ('near_zero', [0,0,1,1], [0.000001,0,1,1]),
]
results = []
for name, p, t in cases:
    pred = torch.tensor([p], dtype=torch.float32, requires_grad=True)
    target = torch.tensor([t], dtype=torch.float32)
    loss = NWD()(pred, target)
    loss.sum().backward()
    delta = [(x-y)*s for x,y,s in zip(p,t,[1,1,.5,.5])]
    distance = math.sqrt(sum(x*x for x in delta))
    expected = -math.expm1(-distance/12.8)
    grad = [math.exp(-distance/12.8)/12.8*x/distance*s if distance else 0. for x,s in zip(delta,[1,1,.5,.5])]
    torch.testing.assert_close(loss, torch.tensor([expected]), atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(pred.grad, torch.tensor([grad]), atol=1e-7, rtol=1e-5)
    assert torch.isfinite(pred.grad).all()
    results.append({'case':name,'loss':loss.item(),'gradient':pred.grad.tolist()})
# FP16 input arithmetic promoted before distance calculation, including overflow-scale centers.
p = torch.tensor([[60000., 0., 2., 2.]], dtype=torch.float16, requires_grad=True)
t = torch.tensor([[-60000., 0., 2., 2.]], dtype=torch.float16)
y = NWD()(p,t); y.sum().backward()
assert y.dtype == torch.float32 and torch.isfinite(y).all() and torch.isfinite(p.grad).all()
e = torch.empty((0,4),requires_grad=True); z=NWD()(e,e.detach());z.sum().backward();assert z.shape==(0,)
x = torch.tensor([[1.,2.,3.,4.],[2.,3.,4.,5.]])
y = x+1
for reduction in ('mean','sum'):
    torch.testing.assert_close(NWD(reduction=reduction)(x,y), getattr(NWD()(x,y),reduction)())
# Exact source identity for all existing default training/inference paths.
paths = ['yolox/models/losses.py','yolox/models/yolo_head.py','yolox/core/trainer.py','yolox/data/data_augment.py','yolox/data/datasets/mosaicdetection.py','yolox/exp/yolox_base.py','exps/example/uavswarm/yolox_x_uavswarmv1_full.py','tools/export_uavswarm_detections.py']
for path in paths:
    expected = subprocess.check_output(['git','-C',str(ROOT),'show',BASE+':'+path])
    assert (ROOT/path).read_bytes() == expected, path
base_scope={};exec(subprocess.check_output(['git','-C',str(ROOT),'show',BASE+':yolox/models/losses.py'],text=True),base_scope)
torch.manual_seed(0)
p=torch.rand(100,4)*10;t=torch.rand(100,4)*10
for kind in ('iou','giou'):
    for reduction in ('none','sum','mean'):
        a=p.clone().requires_grad_();b=p.clone().requires_grad_()
        ya=old(loss_type=kind,reduction=reduction)(a,t)
        yb=base_scope['IOUloss'](loss_type=kind,reduction=reduction)(b,t)
        assert torch.equal(ya,yb)
        ya.sum().backward();yb.sum().backward();assert torch.equal(a.grad,b.grad)
result={'status':'passed','python':sys.version,'torch':torch.__version__,'device':'cpu','nwd_constant':12.8,'cases':results,'fp16_inputs_fp32_arithmetic':True,'old_iou_giou_values_gradients_bitwise_equal':True,'unchanged_base_files':paths}
print(json.dumps(result,indent=2))
