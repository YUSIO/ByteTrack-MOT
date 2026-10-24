"""Finite-budget real detection-loss diagnostic; saves diagnostic-only feedback."""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
import traceback
import gc
from pathlib import Path
import torch
from .model import build
from .data import DetectionPairs


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def state_hash(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def batch(samples, device, variant='correct'):
    x = torch.stack([s['image'] for s in samples]).to(device)
    y = x.new_zeros((len(samples), max(1, max(len(s['targets']) for s in samples)), 5))
    contexts = []
    for i, s in enumerate(samples):
        y[i, :len(s['targets'])] = s['targets'].to(device)
        boxes = s['boxes'].to(device).clone()
        if variant == 'shifted':
            h, w = x.shape[-2:]
            size = boxes[:, 2:]
            centers = boxes[:, :2] + size / 2
            centers = (centers + boxes.new_tensor([w / 2, h / 2])) % boxes.new_tensor([w, h])
            boxes[:, :2] = torch.minimum((centers - size / 2).clamp_min(0), boxes.new_tensor([w, h]) - size)
        contexts.append(None if variant == 'off' or not len(boxes) else {'boxes': boxes, 'ids': s['ids'].to(device)})
    return x, y, contexts


@torch.no_grad()
def evaluate(model, samples, device):
    result = {}
    for variant in ('correct', 'off', 'shifted'):
        losses = []
        components = []
        deltas = []
        for sample in samples:
            x, y, context = batch([sample], device, variant)
            output = model(x, y, context)
            loss = output['total_loss']
            components.append({k: float(v) for k, v in output.items()})
            losses.append(float(loss))
            if variant == 'correct':
                c = context[0]
                delta = model.feedback(x, c['boxes'], c['ids']) - x if c else torch.zeros_like(x)
                deltas.append({'rms': float(delta.square().mean().sqrt()), 'max_abs': float(delta.abs().max())})
        result[variant] = {'mean_loss': sum(losses) / len(losses), 'per_frame_losses': losses,
                           'per_frame_components': components}
        if variant == 'correct':
            result[variant]['deltas'] = deltas
    return result


def measured_parameters(module, active):
    for name, value in module.named_parameters():
        if name.startswith('coordinate.'):
            dim = int(name.split('.')[1])
            yield name, value[active[dim]]
        elif name in ('key.weight', 'value.weight', 'patch_embed.weight', 'to_patch.weight'):
            yield name, value


@torch.no_grad()
def fixed_loss_audit(model, samples, device):
    from .fixed_matching import fixed_matching_loss
    values=[]
    for sample in samples:
        x,y,c=batch([sample],device)
        values.append(float(fixed_matching_loss(model,x,y,c)['total_loss']))
    return {'mean_loss':sum(values)/len(values),'per_frame_losses':values}


def regression_checks(before, after, gates):
    correct, off, shifted = (after[v]['mean_loss'] for v in ('correct', 'off', 'shifted'))
    deltas = before['correct']['deltas'] + after['correct']['deltas']
    return {
        'no_harm': correct / off <= gates['check_final_loss_over_off_max'],
        'correct_history': (shifted - correct) / correct >= gates['check_shifted_loss_margin_relative_min'],
        'bounded_nontrivial_residual': all(
            gates['delta_normalized_rms_min'] <= z['rms'] <= gates['delta_normalized_rms_max']
            and z['max_abs'] <= gates['delta_normalized_maxabs_max'] for z in deltas)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    a = parser.parse_args()
    p = json.loads(a.protocol.read_text()); config = json.loads(a.config.read_text())
    assert digest(a.checkpoint) == p['checkpoint_sha256']
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fit, check = [], []
    for seq in p['sequences']:
        data = DetectionPairs(a.data_root, [seq], config['input_size'])
        fit.extend(data[i - 1] for i in p['fit_frames'])
        check.extend(data[i - 1] for i in p['check_frames'])
    dev=[]
    for seq in p.get('dev_sequences',[]):
        data=DetectionPairs(a.data_root,[seq],config['input_size'])
        dev.extend(data[i-1] for i in p['dev_frames'])
    extra_groups = {}
    for name, spec in p.get('additional_check_groups', {}).items():
        if name in ('fit', 'check', 'dev'):
            raise ValueError('Additional group cannot replace an original group')
        samples = []
        for seq in spec['sequences']:
            data = DetectionPairs(a.data_root, [seq], config['input_size'])
            samples.extend(data[i-1] for i in spec['frames'])
        if not samples:
            raise ValueError('Empty regression group')
        extra_groups[name] = samples
    result = {'protocol_sha256': digest(a.protocol), 'config_sha256': digest(a.config),
              'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'code_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
              'python': sys.version, 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
              'device': str(a.device), 'diagnostic_only': True, 'runs': []}
    result['sample_groups'] = {k: [[s['sequence'], s['frame']] for s in v]
        for k, v in [('fit', fit), ('check', check), ('dev', dev), *extra_groups.items()] if v}
    started = time.monotonic()
    with (a.output / 'steps.jsonl').open('w') as log:
        for seed in p['seeds']:
            for arm, settings in p['arms'].items():
                try:
                    torch.manual_seed(seed)
                    settings=copy.deepcopy(settings)
                    assignment=settings.pop('training_assignment','dynamic')
                    if assignment not in ('dynamic','fixed_off'):
                        raise ValueError('Unknown training assignment')
                    cfg = copy.deepcopy(config); cfg['prompt'].update(settings)
                    model = build(cfg, a.checkpoint).to(a.device).train()
                    model.detector.requires_grad_(False)
                    for layer in model.modules():
                        if isinstance(layer, torch.nn.BatchNorm2d): layer.eval()
                    detector_hash = state_hash(model.detector)
                    active = [set() for _ in range(4)]
                    for s in fit:
                        h, w = cfg['input_size']
                        indices = ((s['boxes'] / torch.tensor([w, h, w, h])).clamp(0, 1) * (model.feedback.bins - 1)).round().long()
                        for d in range(4): active[d].update(indices[:, d].tolist())
                    active = [sorted(x) for x in active]
                    init = {n: v.detach().clone() for n, v in measured_parameters(model.feedback, active)}
                    shadow = copy.deepcopy(model.feedback)
                    optimizer = torch.optim.SGD(model.feedback.parameters(), lr=p['lr'], momentum=p['momentum'], weight_decay=p['weight_decay'])
                    decay = torch.optim.SGD(shadow.parameters(), lr=p['lr'], momentum=p['momentum'], weight_decay=p['weight_decay'])
                    before = {'fit': evaluate(model, fit, a.device), 'check': evaluate(model, check, a.device)}
                    if dev: before['dev']=evaluate(model,dev,a.device)
                    for name, samples in extra_groups.items():
                        before[name] = evaluate(model, samples, a.device)
                    generator = torch.Generator().manual_seed(seed + 1000)
                    row = {'seed': seed, 'arm': arm, 'before': before, 'initial_gradients': {}, 'finite': True,
                           'training_assignment':assignment,'initial_feedback_hash':state_hash(model.feedback)}
                    if p.get('fixed_loss_audit'):
                        row['fixed_loss_before']={k:fixed_loss_audit(model,v,a.device) for k,v in [('fit',fit),('check',check),('dev',dev)] if v}
                    for step in range(p['steps']):
                        if step % (len(fit) // p['batch_size']) == 0:
                            order = torch.randperm(len(fit), generator=generator).tolist()
                        pos = (step * p['batch_size']) % len(fit)
                        x, y, c = batch([fit[i] for i in order[pos:pos + p['batch_size']]], a.device)
                        optimizer.zero_grad(set_to_none=True)
                        if assignment=='fixed_off':
                            from .fixed_matching import fixed_matching_loss
                            loss=fixed_matching_loss(model,x,y,c)['total_loss']
                        else:
                            loss = model(x, y, c)['total_loss']
                        loss.backward()
                        finite = bool(torch.isfinite(loss)) and all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in model.feedback.parameters())
                        if not finite:
                            row['finite'] = False; break
                        if step == 0:
                            for name, param in model.feedback.named_parameters():
                                if name in init:
                                    grad = param.grad[active[int(name.split('.')[1])]] if name.startswith('coordinate.') else param.grad
                                    row['initial_gradients'][name] = float(grad.square().mean().sqrt())
                        optimizer.step()
                        for value in shadow.parameters(): value.grad = torch.zeros_like(value)
                        decay.step()
                        entry = {'seed': seed, 'arm': arm, 'step': step + 1, 'loss': float(loss.detach())}
                        entry['samples']=[[fit[i]['sequence'],fit[i]['frame']] for i in order[pos:pos+p['batch_size']]]
                        log.write(json.dumps(entry) + '\n'); log.flush()
                        if step == 0 or (step + 1) % 20 == 0:
                            print(json.dumps(entry), flush=True)
                    after = {'fit': evaluate(model, fit, a.device), 'check': evaluate(model, check, a.device)}
                    if dev: after['dev']=evaluate(model,dev,a.device)
                    for name, samples in extra_groups.items():
                        after[name] = evaluate(model, samples, a.device)
                    decay_values = dict(measured_parameters(shadow, active))
                    updates = {}
                    for name, param in measured_parameters(model.feedback, active):
                        scale = init[name].square().mean().sqrt().clamp_min(1e-6)
                        updates[name] = float((param.detach() - decay_values[name].detach()).square().mean().sqrt() / scale)
                    f, c = after['fit'], after['check']; gates = p['gates_every_seed_both']
                    diagnostics = {
                        'finite': row['finite'], 'detector_bitwise_unchanged': detector_hash == state_hash(model.detector),
                        'fit_relative_loss_reduction': 1 - f['correct']['mean_loss'] / before['fit']['correct']['mean_loss'],
                        'fit_final_improvement_over_off': 1 - f['correct']['mean_loss'] / f['off']['mean_loss'],
                        'check_final_loss_over_off': c['correct']['mean_loss'] / c['off']['mean_loss'],
                        'fit_shifted_loss_margin_relative': (f['shifted']['mean_loss'] - f['correct']['mean_loss']) / f['correct']['mean_loss'],
                        'check_shifted_loss_margin_relative': (c['shifted']['mean_loss'] - c['correct']['mean_loss']) / c['correct']['mean_loss'],
                        'task_update_relative_rms': updates}
                    delta_list = before['fit']['correct']['deltas'] + before['check']['correct']['deltas'] + f['correct']['deltas'] + c['correct']['deltas']
                    checks = {'finite': diagnostics['finite'], 'detector_unchanged': diagnostics['detector_bitwise_unchanged'],
                              'fit_learning': diagnostics['fit_relative_loss_reduction'] >= gates['fit_relative_loss_reduction_min'],
                              'fit_beats_off': diagnostics['fit_final_improvement_over_off'] >= gates['fit_final_improvement_over_off_min'],
                              'check_no_harm': diagnostics['check_final_loss_over_off'] <= gates['check_final_loss_over_off_max'],
                              'fit_correct_history': diagnostics['fit_shifted_loss_margin_relative'] >= gates['fit_shifted_loss_margin_relative_min'],
                              'check_correct_history': diagnostics['check_shifted_loss_margin_relative'] >= gates['check_shifted_loss_margin_relative_min'],
                              'effective_updates': all(v >= gates['task_update_relative_rms_min'] for v in updates.values()),
                              'initial_upstream_gradients': all(v > 0 for v in row['initial_gradients'].values()),
                              'bounded_nontrivial_residual': all(gates['delta_normalized_rms_min'] <= d['rms'] <= gates['delta_normalized_rms_max'] and d['max_abs'] <= gates['delta_normalized_maxabs_max'] for d in delta_list)}
                    row.update(after=after, diagnostics=diagnostics, gates=checks, passed=all(checks.values()), completed_steps=step + 1)
                    row['group_regression_gates'] = {name: regression_checks(before[name], after[name], gates)
                        for name in ['check', *(['dev'] if dev else []), *extra_groups]}
                    if dev:
                        d=after['dev'];dc=d['correct']['mean_loss'];do=d['off']['mean_loss'];ds=d['shifted']['mean_loss']
                        row['dev_diagnostics']={'loss_over_off':dc/do,'shifted_loss_margin_relative':(ds-dc)/dc}
                        row['dev_regression_gates']={
                            'no_harm':dc/do<=gates['check_final_loss_over_off_max'],
                            'correct_history':(ds-dc)/dc>=gates['check_shifted_loss_margin_relative_min'],
                            'bounded_nontrivial_residual':all(gates['delta_normalized_rms_min']<=z['rms']<=gates['delta_normalized_rms_max'] and z['max_abs']<=gates['delta_normalized_maxabs_max'] for z in before['dev']['correct']['deltas']+d['correct']['deltas'])}
                    if p.get('fixed_loss_audit'):
                        row['fixed_loss_after']={k:fixed_loss_audit(model,v,a.device) for k,v in [('fit',fit),('check',check),('dev',dev)] if v}
                    if p.get('save_diagnostic_feedback'):
                        path=a.output/f'probe_seed{seed}_{arm}.pt'
                        torch.save({'feedback':model.feedback.state_dict(),'feedback_config':cfg,
                            'base_checkpoint_sha256':p['checkpoint_sha256'],'diagnostic_only':True,
                            'seed':seed,'arm':arm,'steps':step+1},path)
                        row['diagnostic_feedback']={'path':path.name,'sha256':digest(path)}
                    if arm in p.get('audit_arms', []):
                        from .history_audit import audit_history
                        row['history_audit'] = {'fit': audit_history(model, fit, a.device), 'check': audit_history(model, check, a.device)}
                    result['runs'].append(row)
                    (a.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
                    print(json.dumps({'seed': seed, 'arm': arm, 'gates': checks, 'diagnostics': diagnostics}), flush=True)
                    del model, shadow, optimizer, decay, init, x, y, loss
                    if str(a.device).startswith('cuda'): torch.cuda.empty_cache()
                except Exception:
                    failure = {'seed': seed, 'arm': arm, 'passed': False, 'error': traceback.format_exc()}
                    result['runs'].append(failure)
                    (a.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
                    print(json.dumps(failure), flush=True)
                    model = shadow = optimizer = decay = init = x = y = loss = None
                    gc.collect()
                    if str(a.device).startswith('cuda'): torch.cuda.empty_cache()
    both = [r for r in result['runs'] if r['arm'] == p.get('candidate_arm','both')]
    result.update(passed=len(both) == len(p['seeds']) and all(r['passed'] for r in both), elapsed_seconds=time.monotonic() - started)
    if dev:
        result['dev_regression_passed']=len(both)==len(p['seeds']) and all(all(r.get('dev_regression_gates',{'missing':False}).values()) for r in both)
    result['all_group_regressions_passed'] = len(both) == len(p['seeds']) and all(
        bool(r.get('group_regression_gates')) and all(all(g.values()) for g in r['group_regression_gates'].values())
        for r in both)
    result['small_scale_candidate_passed'] = result['passed'] and result['all_group_regressions_passed']
    (a.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': result['passed'], 'seconds': result['elapsed_seconds']}), flush=True)

if __name__ == '__main__': main()
