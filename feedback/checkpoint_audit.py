"""Full dev checkpoint interventions, no training and no test-set access."""
import argparse,json,subprocess,time
from pathlib import Path
import torch
from .model import build
from .infer import sha256
from .validate import validate

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--sequences',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prior-mode',choices=['off','online','gt_history'],required=True)
    p.add_argument('--device',default='cuda')
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(0)
    cfg=json.loads(a.config.read_text());state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if a.prior_mode!='off' and 'feedback_config' not in state:
        p.error('Feedback intervention requires trained feedback checkpoint')
    epoch=state.get('epoch');del state
    model=build(cfg,a.checkpoint).to(a.device).eval()
    started=time.monotonic()
    result=validate(model,cfg,None,a.data_root,a.sequences,a.output,a.device,a.prior_mode)
    result.update(checkpoint_sha256=sha256(a.checkpoint),checkpoint_epoch=epoch,
        config_sha256=sha256(a.config),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        torch=str(torch.__version__),device=a.device,seconds=time.monotonic()-started,
        scope='full detector-seen module-held-out dev; explicit read-only intervention; no training')
    (a.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
