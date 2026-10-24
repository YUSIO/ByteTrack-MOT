"""Train-only paired fine-tuning and PSMP pretraining.

Optional train-split closed-loop validation selects best weights by COCO AP.
No test data are loaded here.
"""
import argparse
import json
import random
import time
import subprocess
import copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .data import DetectionPairs, TrajectoryWindows
from .model import build
from .infer import sha256


def configure_detector_training(model, freeze_detector=False):
    """Keep training loss active; freezing weights must not cut input gradients."""
    model.train()
    if freeze_detector:
        if model.mode == 'none':
            raise ValueError('No trainable feedback exists when detector and mode=none are frozen')
        model.detector.requires_grad_(False)
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()


def backward_loss(loss, frozen_detector=False):
    # A frozen detector plus an entirely empty-prior batch has no trainable path.
    # Preserve exact identity and skip updates rather than invent a prompt/gradient.
    if not loss.requires_grad:
        if not frozen_detector:
            raise RuntimeError('Loss has no trainable path')
        return False
    loss.backward()
    return True


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
    p.add_argument('--validation-sequences', nargs='+')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--keep-epochs', action='store_true')
    p.add_argument('--freeze-detector', action='store_true', help='Feedback-only repair; preserve detector weights exactly')
    p.add_argument('--validation-off', action='store_true', help='Paired full dev without feedback every epoch')
    p.add_argument('--validate-initial', action='store_true', help='Record epoch0 online and OFF baselines before any update')
    p.add_argument('--max-steps', type=int, default=0, help='Nonzero labels the entire invocation preflight')
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1 or a.max_steps < 0:
        p.error('Invalid training duration/batch size')
    if a.validation_sequences and (set(a.sequences) & set(a.validation_sequences) or a.phase != 'detector'):
        p.error('Validation must be disjoint train-split sequences for detector training')
    if (a.freeze_detector or a.validation_off or a.validate_initial) and a.phase != 'detector':
        p.error('Detector diagnostics cannot be used for PSMP')
    if (a.validation_off or a.validate_initial) and not a.validation_sequences:
        p.error('Paired validation requires explicit validation sequences')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    config = json.loads(a.config.read_text())
    device = torch.device(a.device)
    predictor = None
    for initialization in [a.checkpoint, a.psmp_checkpoint]:
        if initialization is not None:
            state = torch.load(initialization, map_location='cpu', weights_only=True)
            if state.get('metadata', {}).get('preflight') and not a.max_steps:
                p.error('Preflight weights may only initialize another explicit --max-steps preflight')
            del state
    if a.phase == 'psmp':
        from .psmp import PSMP, psmp_loss
        if a.checkpoint:
            p.error('PSMP resume is not implemented; create a new explicit training attempt')
        model = PSMP(**config['psmp']).to(device)
        data = TrajectoryWindows(a.data_root, a.sequences, config['input_size'],
                                 config['psmp']['observed'], config['psmp']['predicted'])
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
                                        num_workers=a.workers, collate_fn=collate)
    if a.phase == 'detector':
        configure_detector_training(model, a.freeze_detector)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=a.lr, momentum=.9, weight_decay=1e-4)
    detector_hash = None
    if a.freeze_detector:
        from .learning_gate import state_hash
        detector_hash = state_hash(model.detector)
    shadow = shadow_optimizer = initial_feedback = None
    touched = [set() for _ in range(4)]
    if a.freeze_detector and config['mode']=='pp':
        shadow = copy.deepcopy(model.feedback)
        shadow_optimizer = torch.optim.SGD(shadow.parameters(),lr=a.lr,momentum=.9,weight_decay=1e-4)
        initial_feedback = {n:p.detach().clone() for n,p in model.feedback.named_parameters()}
    a.output.mkdir(parents=True, exist_ok=False)
    metadata = {'args': {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                'config': config, 'torch': str(torch.__version__), 'prior_source': 'train_gt_history',
                'preflight': bool(a.max_steps), 'checkpoint_selection': 'closed_loop_dev_AP50_95_earlier_tie' if a.validation_sequences else 'last_not_validation_selected',
                'checkpoint_sha256': sha256(a.checkpoint) if a.checkpoint else None,
                'psmp_sha256': sha256(a.psmp_checkpoint) if a.psmp_checkpoint else None,
                'code_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                'code_dirty': bool(subprocess.check_output(['git','status','--porcelain'], text=True).strip()),
                'samples': len(data), 'status': 'running', 'detector_frozen':a.freeze_detector,
                'initial_detector_state_sha256':detector_hash}
    (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(a.output/'tensorboard'))
    best_ap = -1.0
    validation_rows = []
    started = time.monotonic()
    step = 0
    first_feedback_gradients_recorded = False
    if a.validate_initial:
        from .validate import validate
        initial = {}
        for mode in ['online','off']:
            initial[mode] = validate(model,config,predictor,a.data_root,a.validation_sequences,
                                    a.output/'validation'/('epoch_000_'+mode),device,mode)
        (a.output/'initial_validation.json').write_text(json.dumps(initial,indent=2)+'\n')
    residual_samples = []
    def record_training_residual(module, args, enhanced):
        with torch.no_grad():
            delta=enhanced-args[0]
            residual_samples.append((float(delta.square().mean().sqrt()),float(delta.abs().max())))
    with (a.output/'loss.jsonl').open('w') as log:
        for epoch in range(a.epochs):
            model.train()
            if a.phase == 'detector':
                configure_detector_training(model,a.freeze_detector)
            residual_handle = model.feedback.register_forward_hook(record_training_residual) if a.phase=='detector' and config['mode']=='pp' else None
            for samples in loader:
                residual_samples.clear()
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
                backward_performed = backward_loss(loss,a.freeze_detector)
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite gradient')
                if shadow is not None:
                    for c in contexts:
                        if c is not None:
                            h,w=images.shape[-2:]
                            indices=(c['boxes']/c['boxes'].new_tensor([w,h,w,h])).clamp(0,1)
                            indices=(indices*(model.feedback.bins-1)).round().long().cpu()
                            for d in range(4):touched[d].update(indices[:,d].tolist())
                    if backward_performed and not first_feedback_gradients_recorded:
                        gradients={n:None if p.grad is None else float(p.grad.square().mean().sqrt()) for n,p in model.feedback.named_parameters()}
                        (a.output/'initial_feedback_gradients.json').write_text(json.dumps(gradients,indent=2)+'\n')
                        first_feedback_gradients_recorded = True
                optimizer.step()
                if shadow is not None:
                    for param,reference in zip(model.feedback.parameters(),shadow.parameters()):
                        reference.grad=None if param.grad is None else torch.zeros_like(reference)
                    shadow_optimizer.step()
                writer.add_scalar('train/loss', float(loss.detach()), step+1)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step+1)
                if a.phase == 'detector' and config['mode'] == 'eec':
                    model.feedback.clamp_parameters()
                step += 1
                entry={'epoch':epoch+1,'step':step,'loss':float(loss.detach()),'backward_performed':backward_performed}
                if a.phase=='detector':
                    entry['loss_components']={k:float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in result.items()}
                if residual_samples:
                    entry['feedback']={'count':len(residual_samples),
                        'rms_min':min(v[0] for v in residual_samples),'rms_max':max(v[0] for v in residual_samples),
                        'max_abs':max(v[1] for v in residual_samples)}
                log.write(json.dumps(entry)+'\n');log.flush()
                if step % 50 == 0 or step <= 2:
                    print(json.dumps({'epoch':epoch+1,'step':step,'loss':float(loss.detach()),
                                      'elapsed_seconds':time.monotonic()-started,
                                      'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0}),flush=True)
                if a.max_steps and step >= a.max_steps:
                    break
            if residual_handle is not None:
                residual_handle.remove()
            if a.freeze_detector:
                assert state_hash(model.detector)==detector_hash,'Frozen detector changed'
            if shadow is not None:
                from .learning_gate import measured_parameters
                active=[sorted(x) for x in touched]
                reference=dict(measured_parameters(shadow,active))
                updates={}
                for n,p in measured_parameters(model.feedback,active):
                    init=initial_feedback[n]
                    if n.startswith('coordinate.'):
                        init=init[active[int(n.split('.')[1])]]
                    updates[n]=float((p.detach()-reference[n].detach()).square().mean().sqrt()/init.square().mean().sqrt().clamp_min(1e-6))
                (a.output/f'feedback_learning_epoch_{epoch+1:03d}.json').write_text(json.dumps({
                    'task_update_relative_rms':updates,'active_coordinate_counts':[len(x) for x in active],
                    'decay_reference':'same-device float32 SGD, same gradient-presence mask and steps; zero task gradients',
                    'detector_bitwise_unchanged':state_hash(model.detector)==detector_hash},indent=2)+'\n')
            if a.validation_sequences:
                from .validate import validate
                metrics = validate(model, config, predictor, a.data_root, a.validation_sequences,
                                   a.output/'validation'/f'epoch_{epoch+1:03d}', device)
                ap = metrics['metrics']['AP50:95']
                row={'epoch':epoch+1, **metrics['metrics']}
                row['residuals']=metrics['residuals']
                row['detector_bitwise_unchanged']=state_hash(model.detector)==detector_hash if a.freeze_detector else None
                if a.validation_off:
                    off=validate(model,config,predictor,a.data_root,a.validation_sequences,
                                 a.output/'validation'/f'epoch_{epoch+1:03d}_off',device,'off')
                    row['off_metrics']=off['metrics']
                validation_rows.append(row)
                (a.output/'validation.json').write_text(json.dumps(validation_rows,indent=2)+'\n')
                for key,value in metrics['metrics'].items():
                    writer.add_scalar('validation/'+key, value, epoch+1)
                metadata.update(best_epoch=epoch+1 if ap > best_ap else metadata.get('best_epoch'),
                                best_ap=max(best_ap,ap))
            saved = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch+1,
                     'seed':a.seed, 'metadata':metadata,
                     'psmp_config' if a.phase=='psmp' else 'feedback_config':
                     config['psmp'] if a.phase=='psmp' else config}
            if a.keep_epochs:
                torch.save(saved, a.output/f'epoch_{epoch+1:03d}.pt')
            if a.validation_sequences and ap > best_ap:
                best_ap = ap
                torch.save(saved, a.output/'best.pt')
            torch.save(saved, a.output/'last.pt')
            writer.flush()
            print(json.dumps({'epoch':epoch+1,'steps':step,'loss':float(loss.detach()),'best_ap':best_ap}),flush=True)
            if a.max_steps and step >= a.max_steps:
                break
    writer.close()
    if validation_rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots()
        ax.plot([r['epoch'] for r in validation_rows],[r['AP50:95'] for r in validation_rows])
        ax.set(xlabel='Epoch',ylabel='Detector-seen module-held-out closed-loop AP50:95')
        fig.savefig(a.output/'validation_curve.png',bbox_inches='tight');plt.close(fig)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values=[json.loads(line) for line in (a.output/'loss.jsonl').read_text().splitlines()]
    fig,ax=plt.subplots();ax.plot([r['step'] for r in values],[r['loss'] for r in values],linewidth=.3)
    ax.set(xlabel='Step',ylabel='Training loss');fig.savefig(a.output/'training_curve.png',bbox_inches='tight');plt.close(fig)
    metadata.update(status='completed', steps=step, elapsed_seconds=time.monotonic()-started)
    (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'status':'completed','steps':step,'preflight':bool(a.max_steps)}))


if __name__ == '__main__':
    main()
