"""Fixed train batch, unchanged initial weights: isolate AMP scaling from NWD."""
import json, os, random
from pathlib import Path
import torch
from yolox.exp import get_exp
from yolox.core import Trainer
from yolox.models.losses import IOUloss
from tools.train import make_parser

out=Path(os.environ['EXP198_DIAGNOSTIC_DIR'])
exp_file='exps/example/uavswarm/yolox_x_uavswarmv1_nwd.py'
exp=get_exp(exp_file,None);exp.input_size=(1024,1824);exp.output_dir=str(out)
args=make_parser().parse_args(['-f',exp_file,'-b','4','-d','1','--fp16','-c','/home/yusio/projects/MOT/experiments/exp141_bytetrack_yolox_x_uavswarmv1/pretrained/yolox_x.pth','-expn','model'])
random.seed(0);torch.manual_seed(0)
torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=True
trainer=Trainer(exp,args);trainer.before_train()
inps,targets=trainer.prefetcher.next();inps=inps.to(trainer.data_type);targets=targets[:,:,:5].to(trainer.data_type)
assert tuple(inps.shape)==(4,3,1024,1824)
model=trainer.model;nwd=model.head.iou_loss
buffers={k:v.detach().clone() for k,v in model.named_buffers()}
rows=[]
for method,scale in [('nwd',65536.),('nwd',1.),('iou2',65536.)]:
    with torch.no_grad():
        for name,value in model.named_buffers():value.copy_(buffers[name])
    model.zero_grad(set_to_none=True)
    model.head.iou_loss=nwd if method=='nwd' else IOUloss(reduction='none')
    reg={}
    def inspect(module,args,result):
        reg.update(input_dtype=str(args[0].dtype),loss_dtype=str(result.dtype),loss_finite=bool(torch.isfinite(result).all().item()))
        def grad_hook(g):
            reg.update(input_grad_finite=bool(torch.isfinite(g).all().item()),input_grad_absmax=float(g.abs().max().item()))
        args[0].register_hook(grad_hook)
    handle=model.head.iou_loss.register_forward_hook(inspect)
    with torch.cuda.amp.autocast(enabled=True):outputs=model(inps,targets)
    assert all(torch.isfinite(v).all().item() for k,v in outputs.items() if torch.is_tensor(v)), outputs
    (outputs['total_loss']*scale).backward()
    bad=[];finite_max=0.;count=0
    for name,param in model.named_parameters():
        if param.grad is None:continue
        count+=1
        if not torch.isfinite(param.grad).all().item():bad.append(name)
        else:finite_max=max(finite_max,param.grad.abs().max().item())
    row={'method':method,'scale':scale,'losses':{k:float(v) for k,v in outputs.items()},'parameter_tensors_with_grad':count,'bad_parameter_count':len(bad),'bad_parameters':bad,'finite_grad_absmax':finite_max,'regression':reg}
    rows.append(row);(out/'diagnostic.json').write_text(json.dumps({'status':'observed','input_hw':[1024,1824],'batch':4,'same_train_batch':True,'same_initial_parameters':True,'bn_buffers_restored':True,'optimizer_steps':0,'cases':rows},indent=2)+'\n')
    print(json.dumps(row),flush=True);handle.remove();del outputs
model.head.iou_loss=nwd
print('diagnostic completed',flush=True)
