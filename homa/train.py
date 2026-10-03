import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .data import Windows
from .model import MPANet, association_loss, window_association_loss


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def singleton(batch):
    return batch[0]


def loss_for(model, sample, device):
    c = sample['crops'].to(device, non_blocking=True)
    if model.architecture == 'paper_v2':
        embeddings = model.encode(c)
        return window_association_loss(model, embeddings, sample['all_ids'].to(device),
                                       sample['all_frames'].to(device), model.fused_loss_weight)
    parts, fused = model(c, sample['n_current'])
    return association_loss(parts, fused, *(sample[k].to(device) for k in ('current_ids','history_ids','history_frames')))


def finish_optimizer_step(model, optimizer, scaler, pending, batch_windows):
    """Skip an overflowing AMP accumulation and lower its scale, never step NaNs."""
    scaler.unscale_(optimizer)
    if pending < batch_windows:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.mul_(batch_windows / pending)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
    skipped = not bool(torch.isfinite(norm))
    if skipped:
        if not scaler.is_enabled():
            raise FloatingPointError('nonfinite gradient with AMP disabled')
        # Explicit scale update also handles an overflowing norm with finite grads.
        # No optimizer step: parameters AND momentum buffers stay unchanged.
        scaler.update(new_scale=scaler.get_scale() / 2.)
    else:
        scaler.step(optimizer)
        scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return skipped


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tensorboard', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    cfg = json.loads(args.config.read_text()); t = cfg['train']
    assert not set(t['sequences']) & set(t['validation_sequences'])
    assert torch.cuda.is_available(), 'CUDA required; no silent fallback'
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda:0')
    train = Windows(args.data, t['sequences'], t['window'], t['sampling_stride'])
    val = Windows(args.data, t['validation_sequences'], t['window'], t['validation_stride'])
    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=1, shuffle=shuffle, collate_fn=singleton, num_workers=t['workers'], pin_memory=True, persistent_workers=t['workers'] > 0, generator=torch.Generator().manual_seed(cfg['seed']))
    train_loader, val_loader = loader(train, True), loader(val, False)
    model = MPANet(pretrained=True, architecture=cfg['model'].get('architecture','legacy_v1')).to(device)
    model.fused_loss_weight = t.get('fused_loss_weight', 1.)
    opt = torch.optim.SGD(model.parameters(), lr=t['lr'], momentum=t['momentum'], weight_decay=t['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=t['amp'])
    writer = SummaryWriter(str(args.tensorboard))
    manifest = {'status':'running','phase':'preflight' if args.smoke else 'train','python_executable':sys.executable,'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'config':cfg,'config_sha256':sha(args.config),'data':str(args.data),'train_windows':len(train),'val_windows':len(val),'parameters':sum(p.numel() for p in model.parameters()),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'dirty':subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip(),'seed':cfg['seed'],'initial_weights':{},'dataset_files':{},'amp_overflow_policy':'skip_optimizer_step_halve_scale_fail_after_16_consecutive'}
    for part in (t['sequences'],t['validation_sequences']):
        for s in part:
            for name in ('seqinfo.ini','gt/gt.txt'):
                path=args.data/'train'/s/name
                manifest['dataset_files'][str(path.relative_to(args.data))]=sha(path)
    init=Path(torch.hub.get_dir())/'checkpoints/resnet50-0676ba61.pth'
    if init.exists(): manifest['initial_weights']={'url':'https://download.pytorch.org/models/resnet50-0676ba61.pth','sha256':sha(init)}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('train_windows','val_windows','parameters','code_commit','gpu')}),flush=True)
    overflow_skips=0; consecutive_skips=0
    best=float('inf'); epochs=1 if args.smoke else t['epochs']
    for epoch in range(1, epochs+1):
        start=time.time(); model.train(); opt.zero_grad(set_to_none=True)
        loss_sum=0.; count=0; pending=0; train_limit=min(8,len(train)) if args.smoke else len(train)
        for step,sample in enumerate(train_loader,1):
            with torch.autocast('cuda',enabled=t['amp']):
                loss,n=loss_for(model,sample,device)
            if not torch.isfinite(loss): raise FloatingPointError(f'nonfinite train loss at {epoch}:{step}')
            scaler.scale(loss/t['batch_windows']).backward()
            pending+=1; loss_sum+=float(loss.detach()); count+=1
            if pending==t['batch_windows'] or step==train_limit:
                old_scale = scaler.get_scale()
                skipped = finish_optimizer_step(model, opt, scaler, pending, t['batch_windows'])
                pending = 0
                overflow_skips += int(skipped)
                consecutive_skips = consecutive_skips + 1 if skipped else 0
                if skipped:
                    print(json.dumps({'event':'amp_overflow_skip','epoch':epoch,'window':step,'old_scale':old_scale,'new_scale':scaler.get_scale(),'total_skips':overflow_skips}),flush=True)
                if consecutive_skips >= 16:
                    raise FloatingPointError('16 consecutive AMP overflows; training is unstable')
            if step%100==0: print(json.dumps({'epoch':epoch,'window':step,'loss':loss_sum/count,'elapsed_sec':time.time()-start}),flush=True)
            if step>=train_limit: break
        model.eval(); val_sum=0.; val_n=0
        with torch.inference_mode():
            for step,sample in enumerate(val_loader,1):
                with torch.autocast('cuda',enabled=t['amp']): loss,n=loss_for(model,sample,device)
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite validation loss')
                val_sum+=float(loss); val_n+=1
                if args.smoke and step>=2: break
        row={'epoch':epoch,'train_loss':loss_sum/count,'val_loss':val_sum/val_n,'seconds':time.time()-start,'max_cuda_gib':torch.cuda.max_memory_allocated()/1024**3,'amp_overflow_skips':overflow_skips,'loss_scale':scaler.get_scale()}
        print(json.dumps(row),flush=True)
        with (args.output/'curves.csv').open('a') as f:
            w=csv.DictWriter(f,fieldnames=row.keys())
            if epoch==1:w.writeheader()
            w.writerow(row)
        for k in ('train_loss','val_loss','max_cuda_gib'): writer.add_scalar(k,row[k],epoch)
        state={'model':model.state_dict(),'epoch':epoch,'config':cfg,'val_loss':row['val_loss'],'code_commit':manifest['code_commit']}
        if not args.smoke:
            torch.save(state,args.output/'last.pt')
            if row['val_loss']<best:
                best=row['val_loss']; torch.save(state,args.output/'best.pt'); manifest['best_epoch']=epoch; manifest['best_val_loss']=best
        writer.flush()
    writer.close()
    manifest.update(status='completed',finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),epochs_completed=epochs)
    manifest['weights']={p.name:sha(p) for p in args.output.glob('*.pt')}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__': main()
