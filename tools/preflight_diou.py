"""Bounded maximum-resolution training probe; never a formal training run."""
import json, os, random, sys
from pathlib import Path
import torch
from yolox.exp import get_exp
from yolox.core import Trainer
from tools.train import make_parser
batch = int(sys.argv[1])
root = Path(os.environ['EXP199_ROOT'])
exp_file = 'exps/example/uavswarm/yolox_x_uavswarmv1_diou.py'
exp = get_exp(exp_file, None)
exp.input_size = (1024, 1824)
exp.output_dir = str(root / 'preflight')
args = make_parser().parse_args(['-f',exp_file,'-b',str(batch),'-d','1','--fp16','-c','/home/yusio/projects/MOT/experiments/exp141_bytetrack_yolox_x_uavswarmv1/pretrained/yolox_x.pth','-expn',f'batch{batch}'])
random.seed(0); torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
steps=[]
trainer = Trainer(exp,args)
trainer.before_train()
assert trainer.model.head.iou_loss.loss_type == "diou"
trainer.epoch = 0
successful_steps = 0
for i in range(20):
    trainer.iter = i
    scale_before = trainer.scaler.get_scale()
    trainer.train_one_iter()
    values = {k:float(v.latest) for k,v in trainer.meter.get_filtered_meter('loss').items()}
    assert values and all(torch.isfinite(torch.tensor(x)) for x in values.values()), values
    finite_gradients = all(p.grad is None or torch.isfinite(p.grad).all() for p in trainer.model.parameters())
    scale_after = trainer.scaler.get_scale()
    assert finite_gradients or scale_after < scale_before, "nonfinite gradients without AMP skip"
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
    successful_steps += int(finite_gradients)
    values.update(finite_gradients=bool(finite_gradients), scale_before=scale_before, scale_after=scale_after)
    steps.append(values)
    print(json.dumps({'step':i,'losses':values}),flush=True)
    if successful_steps == 3: break
assert successful_steps == 3, "AMP failed to reach three finite-gradient steps"
torch.cuda.synchronize()
result = {'batch':batch,'input_size':exp.input_size,'max_memory_allocated':torch.cuda.max_memory_allocated(),'max_memory_reserved':torch.cuda.max_memory_reserved(),'status':'passed','optimizer_steps':3,'losses':steps,'cudnn_benchmark':True,'cudnn_deterministic':True}
(root/'preflight'/f'batch{batch}_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
