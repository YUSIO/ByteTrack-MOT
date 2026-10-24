"""Train-only paired fine-tuning and PSMP pretraining.

This executable saves last/epoch states, not validation-selected best weights.
Formal selection/evaluation is a separate run. No test data are loaded here.
"""
import argparse
import json
import random
import subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .data import DetectionPairs, TrajectoryWindows
from .model import build
from .infer import sha256


def collate(samples):
    return samples


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--phase', choices=['detector', 'psmp'], required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--psmp-checkpoint', type=Path)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--sequences', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, required=True)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-steps', type=int, default=0, help='Nonzero labels the entire invocation preflight')
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1 or a.max_steps < 0:
        p.error('Invalid training duration/batch size')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    config = json.loads(a.config.read_text())
    device = torch.device(a.device)
    predictor = None
    if a.phase == 'psmp':
        from .psmp import PSMP, psmp_loss
        if a.checkpoint:
            p.error('PSMP resume is not implemented; create a new explicit training attempt')
        model = PSMP(**config['psmp']).to(device)
        data = TrajectoryWindows(a.data_root, a.sequences, config['input_size'])
    else:
        if a.checkpoint is None:
            p.error('Detector fine-tuning requires an explicit shared initialization')
        model = build(config, a.checkpoint).to(device)
        data = DetectionPairs(a.data_root, a.sequences, config['input_size'], config.get('window',8))
        if config['mode'] == 'eec':
            from .psmp import load_psmp
            if a.psmp_checkpoint is None:
                p.error('Pretrained PSMP checkpoint is required; no random predictor fallback')
            predictor = load_psmp(a.psmp_checkpoint, config['psmp']).to(device).eval().requires_grad_(False)
    if not len(data):
        raise ValueError('No training samples')
    loader = torch.utils.data.DataLoader(data, batch_size=a.batch_size, shuffle=True,
                                        num_workers=0, collate_fn=collate)
    optimizer = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=.9, weight_decay=1e-4)
    a.output.mkdir(parents=True, exist_ok=False)
    metadata = {'args': {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                'config': config, 'torch': str(torch.__version__), 'prior_source': 'train_gt_history',
                'preflight': bool(a.max_steps), 'checkpoint_selection': 'last_not_validation_selected',
                'checkpoint_sha256': sha256(a.checkpoint) if a.checkpoint else None,
                'psmp_sha256': sha256(a.psmp_checkpoint) if a.psmp_checkpoint else None,
                'code_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                'code_dirty': bool(subprocess.check_output(['git','status','--porcelain'], text=True).strip()),
                'samples': len(data), 'status': 'running'}
    (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    step = 0
    with (a.output/'loss.jsonl').open('w') as log:
        for epoch in range(a.epochs):
            model.train()
            if a.phase == 'detector':
                # Shared small-batch migration recipe: fixed BN stats, train affine weights.
                for layer in model.modules():
                    if isinstance(layer, nn.BatchNorm2d):
                        layer.eval()
            for samples in loader:
                optimizer.zero_grad(set_to_none=True)
                if a.phase == 'psmp':
                    losses = []
                    for sample in samples:
                        prediction = model(sample['history'][None].to(device))
                        losses.append(psmp_loss(prediction, sample['future'][None].to(device),
                                                sample['reference'][None].to(device))['total_loss'])
                    loss = torch.stack(losses).mean()
                else:
                    images = torch.stack([s['image'] for s in samples]).to(device)
                    previous = torch.stack([s['previous'] for s in samples]).to(device)
                    targets = images.new_zeros((len(samples), max(1,max(len(s['targets']) for s in samples)),5))
                    contexts = []
                    for i,s in enumerate(samples):
                        targets[i,:len(s['targets'])] = s['targets'].to(device)
                        context = None
                        if config['mode'] == 'pp' and len(s['ids']):
                            context = {'boxes':s['boxes'].to(device), 'ids':s['ids'].to(device)}
                        elif config['mode'] == 'eec' and len(s['history']):
                            with torch.no_grad():
                                prior = predictor(s['history'][None].to(device))['prior']
                            context = {'prior':prior}
                        contexts.append(context)
                    result = model(images, targets, contexts,
                                   previous if config['mode']=='eec' else None)
                    loss = result['total_loss']
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite gradient')
                optimizer.step()
                if a.phase == 'detector' and config['mode'] == 'eec':
                    model.feedback.clamp_parameters()
                step += 1
                log.write(json.dumps({'epoch':epoch+1,'step':step,'loss':float(loss.detach())})+'\n');log.flush()
                if a.max_steps and step >= a.max_steps:
                    break
            saved = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch+1,
                     'seed':a.seed, 'metadata':metadata,
                     'psmp_config' if a.phase=='psmp' else 'feedback_config':
                     config['psmp'] if a.phase=='psmp' else config}
            torch.save(saved, a.output/f'epoch_{epoch+1:03d}.pt')
            torch.save(saved, a.output/'last.pt')
            if a.max_steps and step >= a.max_steps:
                break
    metadata.update(status='completed', steps=step)
    (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'status':'completed','steps':step,'preflight':bool(a.max_steps)}))


if __name__ == '__main__':
    main()
