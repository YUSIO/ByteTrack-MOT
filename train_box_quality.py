"""Fixed-budget training on sequence-disjoint train/dev real runtime proposals."""
from pathlib import Path
import argparse,datetime,hashlib,json,platform,subprocess,sys,time,traceback
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from box_quality import QualityNet

def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--run',required=True);args=ap.parse_args()
    E=args.root/'experiments/exp158_single_frame_box_quality';W=Path(__file__).resolve().parent;R=E/'results'/args.run
    git=lambda *a:subprocess.check_output(['git','-C',str(W),*a],text=True).strip()
    assert not git('status','--porcelain')
    cfg=json.loads((E/'configs/train.json').read_text());split=json.loads((E/'configs/split_source.json').read_text());assert not set(split['fit_sequences'])&set(split['dev_sequences'])
    R.mkdir(parents=True,exist_ok=False);(R/'command.txt').write_text(' '.join(sys.argv)+'\n')
    manifest=dict(status='running',phase='train',commit=git('rev-parse','HEAD'),branch='exp/158-box-quality',private_repository='git@github.com:YUSIO/ByteTrack-MOT.git',clean=True,python_executable=sys.executable,python=sys.version,torch=torch.__version__,numpy=np.__version__,host=platform.node(),device='cuda:1',seed=158,config=cfg,config_sha256=sha(E/'configs/train.json'),split_sha256=sha(E/'configs/split_source.json'),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    dump(R/'manifest.yaml',manifest);rc=0
    try:
        torch.set_num_threads(4);torch.manual_seed(158);np.random.seed(158);torch.cuda.set_device(1)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        inputs={};sets={}
        for role,key in [('fit','fit_sequences'),('dev','dev_sequences')]:
            packs=[]
            for name in split[key]:
                p=E/'results/run_001/collect'/f'{name}.pairs.npz';inputs[str(p.relative_to(args.root))]=sha(p)
                with np.load(p) as z:packs.append({k:z[k] for k in ['images','boxes','labels','missing']})
            sets[role]={k:torch.from_numpy(np.concatenate([p[k] for p in packs])) for k in packs[0]}
        dump(R/'inputs.json',inputs)
        model=QualityNet().cuda(1);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);writer=SummaryWriter(str(R/'tensorboard'));history=[];best=float('inf')
        print('samples', {k:len(v['images']) for k,v in sets.items()},'parameters',sum(p.numel() for p in model.parameters()),flush=True)
        for epoch in range(1,cfg['epochs']+1):
            start=time.time();model.train();trainloss=0;fit=sets['fit'];order=torch.randperm(len(fit['images']))
            for ids in order.split(cfg['batch_size']):
                image=fit['images'][ids].cuda(1).float()/255;boxes=fit['boxes'][ids].cuda(1);target=fit['labels'][ids].cuda(1)
                pred=model(image,boxes);loss=torch.nn.functional.smooth_l1_loss(pred,target,beta=.1)
                optimizer.zero_grad();loss.backward();optimizer.step();trainloss+=loss.item()*len(ids)
            model.eval();preds=[]
            with torch.no_grad():
                for ids in torch.arange(len(sets['dev']['images'])).split(cfg['batch_size']):
                    preds.append(model(sets['dev']['images'][ids].cuda(1).float()/255,sets['dev']['boxes'][ids].cuda(1)).cpu())
            q=torch.cat(preds);y=sets['dev']['labels'];missing=sets['dev']['missing'];q0=torch.where(missing,torch.zeros_like(q[:,0]),q[:,0]);y0=torch.where(missing,torch.zeros_like(y[:,0]),y[:,0]);choose=(q[:,1]>=.5)&(q[:,1]>q0);actual=torch.where(choose,y[:,1],y0)
            oracle=torch.where((y[:,1]>=.5)&(y[:,1]>y0),y[:,1],y0);regret=(oracle-actual).mean().item();mae=(q-y).abs().mean().item()
            row=dict(epoch=epoch,train_loss=trainloss/len(fit['images']),dev_mae=mae,dev_selection_regret=regret,dev_iou_gain=(actual-y0).mean().item(),dev_oracle_iou_gain=(oracle-y0).mean().item(),seconds=time.time()-start)
            history.append(row);print(json.dumps(row),flush=True)
            for k,v in row.items():writer.add_scalar(k,v,epoch)
            torch.save(model.state_dict(),R/'last.pt')
            if regret<best:best=regret;torch.save(model.state_dict(),R/'best.pt');dump(R/'selection.json',row)
            dump(R/'history.json',history)
        writer.close();dump(R/'metrics.json',dict(selected=json.loads((R/'selection.json').read_text()),parameters=sum(p.numel() for p in model.parameters()),samples={k:len(v['images']) for k,v in sets.items()},best_sha256=sha(R/'best.pt'),last_sha256=sha(R/'last.pt')))
    except Exception:traceback.print_exc();rc=1
    manifest.update(status='completed' if rc==0 else 'failed',exit_code=rc,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());dump(R/'manifest.yaml',manifest);(R/'exit_code.txt').write_text(str(rc)+'\n');(R/'report.md').write_text('# 训练记录\n\n'+('完成；按开发集候选选择 regret 最小选权重，待闭环评估。' if rc==0 else '失败，保留证据。')+'\n');sys.exit(rc)
if __name__=='__main__':main()
