import importlib.util, json, torch
s=importlib.util.spec_from_file_location('losses','yolox/models/losses.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
p=torch.tensor([[0.,0.,2.,2.],[1.,0.,2.,2.],[0.,0.,1.,1.],[4.,0.,2.,2.],[.01,0.,.1,.1]],requires_grad=True)
t=torch.tensor([[0.,0.,2.,2.],[0.,0.,2.,2.],[0.,0.,2.,2.],[0.,0.,2.,2.],[0.,0.,.1,.1]])
y=m.IOUloss(loss_type='diou')(p,t)
expected=torch.tensor([0.,1-1/3+1/13,.75,1+16/40,1-.009/.011+.0001/.0221])
assert torch.allclose(y,expected,atol=1e-6),(y,expected)
y.sum().backward(); assert torch.isfinite(p.grad).all()
# Existing default path remains exact 1-IoU squared.
z=m.IOUloss()(p.detach(),t);assert torch.allclose(z,1-torch.tensor([1.,1/3,.25,0.,.009/.011])**2,atol=1e-6)
q=p.detach().double().requires_grad_()
# Numerical gradients on non-boundary geometry (DIoU internally uses float32).
q=torch.tensor([[.3,.4,1.1,1.3]],requires_grad=True);g=torch.tensor([[0.,0.,2.,2.]])
f=m.IOUloss(loss_type='diou');a=f(q,g);a.sum().backward()
for i in range(4):
 plus=q.detach().clone();minus=plus.clone();plus[0,i]+=.001;minus[0,i]-=.001
 numeric=(f(plus,g)-f(minus,g))/.002
 assert torch.allclose(numeric,q.grad[0,i],atol=1e-4),(i,numeric,q.grad)
assert torch.allclose(f(p.detach().half(),t.half()),y,atol=2e-4)
print(json.dumps({'status':'passed','diou':y.tolist(),'cases':['identical','partial','contained','disjoint','tiny'],'default_iou2_unchanged':True,'finite_gradients':True,'numerical_gradient_check':True}))
