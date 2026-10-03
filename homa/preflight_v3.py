"""Bounded train-only learnability gate through the actual AMP/accumulation path."""
import argparse
import gc
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from .data import Windows
from .model import MPANet
from .train import loss_for, finish_optimizer_step, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.config.read_text()); t = cfg['train']
    assert cfg['model']['architecture'] == 'residual_norm_v3'
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    samples = []
    for name in t['sequences']:
        ds = Windows(a.data, [name], t['window'], t['sampling_stride'])
        s = ds[0]
        if s['n_current'] >= 3:
            samples.append(s)
        if len(samples) == 3:
            break
    assert len(samples) == 3
    meta = {'scope':'train-only preflight; no checkpoint saved; no validation or test',
            'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'config_sha256':sha(a.config),'torch':torch.__version__,
            'gpu':torch.cuda.get_device_name(), 'amp':t['amp'], 'batch_windows':t['batch_windows'],
            'selection':'first eligible window of first three train sequences with >=3 current objects',
            'samples':[{'sequence':s['sequence'],'frame':s['frame'],'current_objects':s['n_current']} for s in samples],
            'gate':'each seed: final mean loss <= 0.8 * initial; each window final loss < initial; finite gradients; <=2 overflow skips',
            'steps_per_seed':32,'seeds':[42,43],'results':[]}
    (a.output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    for seed in meta['seeds']:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model = MPANet(pretrained=True, architecture=cfg['model']['architecture']).cuda()
        model.fused_loss_weight = t['fused_loss_weight']
        opt = torch.optim.SGD(model.parameters(), lr=t['lr'], momentum=t['momentum'], weight_decay=t['weight_decay'])
        scaler = torch.amp.GradScaler('cuda', enabled=t['amp'])
        rows = []; skips = 0
        for step in range(33):
            if step in (0,8,16,32):
                model.eval()
                with torch.no_grad(), torch.autocast('cuda', enabled=t['amp']):
                    values = [float(loss_for(model,s,'cuda')[0]) for s in samples]
                rows.append({'updates':step,'window_losses':values,'mean_loss':sum(values)/len(values),'overflow_skips':skips})
                print(json.dumps({'seed':seed,**rows[-1]}),flush=True)
            if step == 32:
                break
            model.train(); opt.zero_grad(set_to_none=True)
            for micro in range(t['batch_windows']):
                s = samples[(step*t['batch_windows']+micro)%len(samples)]
                with torch.autocast('cuda',enabled=t['amp']):
                    loss,n = loss_for(model,s,'cuda')
                assert n > 0 and torch.isfinite(loss)
                scaler.scale(loss/t['batch_windows']).backward()
            skips += int(finish_optimizer_step(model,opt,scaler,t['batch_windows'],t['batch_windows']))
        passed = (rows[-1]['mean_loss'] <= .8*rows[0]['mean_loss'] and skips<=2
                  and all(b<a for a,b in zip(rows[0]['window_losses'], rows[-1]['window_losses'])))
        meta['results'].append({'seed':seed,'rows':rows,'passed':passed})
        (a.output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
        del model,opt,scaler; gc.collect(); torch.cuda.empty_cache()
        if not passed:
            raise RuntimeError(f'learnability gate failed seed {seed}')
    meta['passed'] = True
    (a.output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__ == '__main__': main()
