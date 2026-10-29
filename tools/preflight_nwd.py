"""Maximum-resolution probe with bounded native AMP backoff and three valid updates."""
import json, os, random, subprocess, sys
from pathlib import Path
import torch
from yolox.exp import get_exp
from yolox.core import Trainer
from yolox.models.nwd_loss import NWDloss
from tools.train import make_parser

batch=int(sys.argv[1]);out=Path(os.environ['EXP198_PREFLIGHT_DIR'])
assert not (out/f'batch{batch}_result.json').exists()
exp_file='exps/example/uavswarm/yolox_x_uavswarmv1_nwd.py'
exp=get_exp(exp_file,None);exp.input_size=(1024,1824);exp.output_dir=str(out)
args=make_parser().parse_args(['-f',exp_file,'-b',str(batch),'-d','1','--fp16','-c','/home/yusio/projects/MOT/experiments/exp141_bytetrack_yolox_x_uavswarmv1/pretrained/yolox_x.pth','-expn',f'batch{batch}'])
random.seed(0);torch.manual_seed(0)
torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=True
trainer=Trainer(exp,args);trainer.before_train()
assert isinstance(trainer.model.head.iou_loss,NWDloss)
assert trainer.model.head.iou_loss.constant==12.8
trainer.epoch=0
accepted=[0]
def before_optimizer_step(optimizer,args,kwargs):
    grads=[p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
    assert grads and all(torch.isfinite(g).all().item() for g in grads), 'optimizer received nonfinite gradients'
    assert any(torch.count_nonzero(g).item() for g in grads), 'all gradients zero'
def after_optimizer_step(optimizer,args,kwargs):
    accepted[0]+=1
trainer.optimizer.register_step_pre_hook(before_optimizer_step)
trainer.optimizer.register_step_post_hook(after_optimizer_step)
attempts=[];consecutive=0
# Bound calibration. Keep the native scaler initial value and backoff unchanged.
for i in range(20):
    trainer.iter=i
    scale_before=trainer.scaler.get_scale();count_before=accepted[0]
    trainer.train_one_iter()
    scale_after=trainer.scaler.get_scale()
    losses={k:float(v.latest) for k,v in trainer.meter.get_filtered_meter('loss').items()}
    assert losses and all(torch.isfinite(torch.tensor(v)) for v in losses.values()), losses
    bad=[name for name,p in trainer.model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all().item()]
    stepped=accepted[0]==count_before+1
    row={'attempt':i,'losses':losses,'scale_before':scale_before,'scale_after':scale_after,'optimizer_step_executed':stepped,'bad_gradient_parameters':bad}
    attempts.append(row)
    (out/'attempts.json').write_text(json.dumps(attempts,indent=2)+'\n')
    print(json.dumps(row),flush=True)
    if stepped:
        assert not bad and scale_after>=scale_before
        consecutive+=1
    else:
        assert accepted[0]==count_before and bad and scale_after<scale_before, 'unexpected skipped step'
        consecutive=0
    assert all(torch.isfinite(p).all().item() for p in trainer.model.parameters()), 'nonfinite model parameter'
    if consecutive==3:break
assert consecutive==3, 'native AMP did not reach three consecutive finite optimizer updates within 20 attempts'
torch.cuda.synchronize()
result={'status':'passed','batch':batch,'input_size':exp.input_size,'max_memory_allocated':torch.cuda.max_memory_allocated(),'max_memory_reserved':torch.cuda.max_memory_reserved(),'optimizer_steps':accepted[0],'consecutive_finite_optimizer_steps':consecutive,'attempts':attempts,'native_amp_scaler_unchanged':True,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'formal_initialization':'reload_original_COCO_checkpoint','cudnn_benchmark':True,'cudnn_deterministic':True}
(out/f'batch{batch}_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
