"""Bounded maximum-resolution training probe; never a formal training run."""
import json, os, random, sys
from pathlib import Path
import torch
from yolox.exp import get_exp
from yolox.core import Trainer
from tools.train import make_parser
batch = int(sys.argv[1])
root = Path(os.environ['EXP198_ROOT'])
exp_file = 'exps/example/uavswarm/yolox_x_uavswarmv1_nwd.py'
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
from yolox.models.nwd_loss import NWDloss
assert isinstance(trainer.model.head.iou_loss, NWDloss)
assert trainer.model.head.iou_loss.constant == 12.8
trainer.epoch = 0
for i in range(3):
    trainer.iter = i
    trainer.train_one_iter()
    values = {k:float(v.latest) for k,v in trainer.meter.get_filtered_meter('loss').items()}
    assert values and all(torch.isfinite(torch.tensor(x)) for x in values.values()), values
    grads = [p.grad for p in trainer.model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all().item() for g in grads), "nonfinite gradient"
    assert any(torch.count_nonzero(g).item() for g in grads), "all gradients zero"
    steps.append(values)
    print(json.dumps({'step':i,'losses':values}),flush=True)
torch.cuda.synchronize()
result = {'batch':batch,'input_size':exp.input_size,'max_memory_allocated':torch.cuda.max_memory_allocated(),'max_memory_reserved':torch.cuda.max_memory_reserved(),'status':'passed','optimizer_steps':3,'gradients_finite':True,'losses':steps,'cudnn_benchmark':True,'cudnn_deterministic':True}
(root/'preflight'/f'batch{batch}_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
