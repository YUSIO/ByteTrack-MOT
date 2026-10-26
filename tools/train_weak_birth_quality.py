"""Exp181: fixed-budget, scene-held-out quality learning on frozen Seed boxes."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import traceback

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import numpy as np
import torch
from fctrack.verifier import load


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def git(*args):
    return subprocess.check_output(['git', '-C', str(SOURCE), *args], text=True).strip()


def sample_weights(rows):
    """Split positive GT-frame duplicates, equalize sequences, then classes."""
    key = lambda z: (z['sequence'], z['frame'], z['owner'])
    counts = Counter(key(z) for z in rows if int(z['label']))
    weight = np.array([1 / counts[key(z)] if int(z['label']) else 1 for z in rows], float)
    seq = np.array([z['sequence'] for z in rows])
    labels = np.array([int(z['label']) for z in rows])
    assert set(labels) == {0, 1}
    for s in sorted(set(seq)):
        weight[seq == s] /= weight[seq == s].sum()
    for label in [0, 1]:
        weight[labels == label] /= weight[labels == label].sum()
    return weight / weight.sum()


def train_mode(model):
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def bn_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_buffers()
            if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}


@torch.no_grad()
def predict(model, images, scalars, indices, device, batch_size):
    model.eval()
    result = []
    for start in range(0, len(indices), batch_size):
        ids = indices[start:start + batch_size]
        logits = model(images[ids].to(device), scalars[ids].to(device))[:, 0]
        # Match the original verifier's float64 sigmoid.
        logits = logits.cpu().numpy().astype(float)
        result.append(1 / (1 + np.exp(-logits)))
    return np.concatenate(result)


def score_metrics(rows, prediction, threshold):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.array([int(z['label']) for z in rows])
    accepted = prediction >= threshold
    positive = accepted & (y == 1)
    unique = {(z['sequence'], z['frame'], z['owner']) for i, z in enumerate(rows) if positive[i]}
    scenes = sorted({z['scene'] for i, z in enumerate(rows) if positive[i]})
    return dict(AP=float(average_precision_score(y, prediction)),
                AUC=float(roc_auc_score(y, prediction)), accepted=int(accepted.sum()),
                true_positive=int(positive.sum()), false_positive=int((accepted & (y == 0)).sum()),
                precision=float(positive.sum() / accepted.sum()) if accepted.any() else 0.,
                positive_unique_gt_frames=len(unique), positive_scenes=len(scenes), scene_keys=scenes)


def gpu_snapshot(device, expected_uuid, require_idle=True):
    """Refuse an occupied physical GPU; never inspect/alter another process."""
    inventory = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    processes = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader,nounits'], text=True)
    rows = list(csv.reader(inventory.splitlines()))
    target = next([v.strip() for v in row] for row in rows if int(row[0]) == device.index)
    assert target[1] == expected_uuid, target
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '', '0,1'), 'Unexpected GPU remapping'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') != '', 'CUDA hidden'
    if require_idle:
        assert expected_uuid not in processes and int(target[3]) < 100 and int(target[4]) == 0, (target, processes)
    return dict(gpus=inventory, compute_processes=processes, target_uuid=expected_uuid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', default='run_002')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    exp = root / 'experiments/exp181_weak_birth_quality'
    cfg = read(exp / 'configs/train_oof.json')
    source = read(exp / cfg['source_record'])
    inputs = read(exp / cfg['input_record'])
    assert not git('status', '--porcelain') and git('rev-parse', 'HEAD') == source['commit']
    for path, digest in inputs.items():
        assert sha(root / path) == digest, path
    collection = exp / 'results' / cfg['collection_run']
    assert read(collection / 'manifest.yaml')['status'] == 'completed'
    assert (collection / 'exit_code.txt').read_text().strip() == '0'
    assert read(collection / 'metrics.json')['passed']
    for path, digest in read(collection / 'output_sha256.json').items():
        assert sha(collection / path) == digest, path
    rows = list(csv.DictReader((collection / 'samples.csv').open()))
    arrays = np.load(collection / 'crops.npz')
    x = torch.from_numpy(arrays['images'])
    sc = torch.from_numpy(arrays['scalars'])
    y = np.array([int(z['label']) for z in rows])
    folds = np.array([int(z['fold']) for z in rows])
    residual = np.array([bool(int(z['residual'])) for z in rows])
    scenes = np.array([z['scene'] for z in rows])
    assert len(rows) == len(x) == 3975 and x.shape[1:] == (3, 64, 64)
    assert sc.shape == (len(rows), 2) and x.dtype == sc.dtype == torch.float32
    assert torch.isfinite(x).all() and torch.isfinite(sc).all() and x.min() >= 0 and x.max() <= 1
    assert residual.sum() == 2902 and y[residual].sum() == 69
    assert cfg['models'] == ['R', 'F'] and cfg['folds'] == [0, 1, 2]
    assert not any(cfg[key] for key in ['cal_read', 'dev_read', 'test_read'])
    fold_info = {}
    for name in cfg['models']:
        for fold in cfg['folds']:
            tr = (folds != fold) & (residual if name == 'R' else True)
            te = (folds == fold) & residual
            assert set(scenes[tr]).isdisjoint(scenes[te])
            assert set(y[tr]) == set(y[te]) == {0, 1}
            fold_info[f'{name}_{fold}'] = dict(train_rows=int(tr.sum()), train_positive=int(y[tr].sum()),
                train_scenes=sorted(set(scenes[tr])), heldout_rows=int(te.sum()),
                heldout_positive=int(y[te].sum()), heldout_scenes=sorted(set(scenes[te])))
    import sklearn
    from torch.utils.tensorboard import SummaryWriter
    device = torch.device(cfg['device'])
    assert device == torch.device('cuda:1') and torch.cuda.is_available()
    gpu = gpu_snapshot(device, cfg['physical_gpu_uuid'])
    torch.set_num_threads(1)
    torch.manual_seed(cfg['seed'])
    initial = load('seed', root / cfg['pretrained_weights'], torch.device('cpu'))
    reference = np.array([float(z['p_reg']) for z in rows])
    cached_reference = predict(initial, x, sc, np.arange(len(rows)), 'cpu', cfg['batch_size'])
    parity = float(np.max(np.abs(reference - cached_reference)))
    assert parity <= cfg['reference_parity_atol'], parity
    if args.check_only:
        print(json.dumps(dict(input_files=len(inputs), folds=fold_info, cached_reference_max_error=parity,
                              source_clean=True, gpu=gpu), indent=2))
        return 0
    # A fresh read directly before allocation: a prior preflight is not a reservation.
    gpu = gpu_snapshot(device, cfg['physical_gpu_uuid'])
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    run = exp / 'results' / args.run
    run.mkdir(parents=True, exist_ok=False)
    manifest = dict(run=args.run, phase='quality_learning_oof', status='running', source=source,
        dirty=False, device=str(device), physical_gpu_uuid=cfg['physical_gpu_uuid'], threads=1,
        seed=cfg['seed'], python_executable=sys.executable, python=sys.version, host=platform.node(),
        versions=dict(torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__,
                      cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version()),
        config_sha256=sha(exp / 'configs/train_oof.json'), inputs_sha256=sha(exp / cfg['input_record']),
        script_sha256=sha(Path(__file__)), collection_run=cfg['collection_run'],
        cal_read=False, dev_read=False, test_read=False, original_seed_updated=False,
        cached_reference_max_error=parity, deterministic_algorithms=True, tf32=False,
        environment={k: os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']},
        start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    save(run / 'manifest.yaml', manifest)
    save(run / 'frozen_config.json', cfg)
    save(run / 'inputs.json', inputs)
    save(run / 'folds.json', fold_info)
    save(run / 'gpu_preflight.json', gpu)
    (run / 'command.txt').write_text(shlex.join([sys.executable, *sys.argv]) + '\n')
    (run / 'models').mkdir()
    (run / 'sampling').mkdir()
    out = open(run / 'stdout.log', 'w', buffering=1)
    err = open(run / 'stderr.log', 'w', buffering=1)
    os.dup2(out.fileno(), 1)
    os.dup2(err.fileno(), 2)
    rc, writer = 1, None
    try:
        # Cache only on the allocated device. Labels/GT identifiers never enter inference.
        images, scalars = x.to(device), sc.to(device)
        targets = torch.tensor(y, dtype=torch.float32, device=device)
        pred = {name: np.full(len(rows), np.nan) for name in cfg['models']}
        history_file = (run / 'history.csv').open('w', buffering=1)
        history = csv.DictWriter(history_file, fieldnames=['model', 'fold', 'epoch', 'steps', 'loss', 'gradient_norm', 'seconds'])
        history.writeheader()
        for name in cfg['models']:
            for fold in cfg['folds']:
                seed = cfg['seed'] + fold
                torch.manual_seed(seed)
                rng = np.random.default_rng(seed)
                train_ids = np.flatnonzero((folds != fold) & (residual if name == 'R' else True))
                test_ids = np.flatnonzero((folds == fold) & residual)
                weight = sample_weights([rows[i] for i in train_ids])
                model = load('seed', root / cfg['pretrained_weights'], device)
                original_bn = bn_state(model)
                optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
                writer = SummaryWriter(str(run / 'tb' / f'{name}_{fold}'))
                all_ids, all_flips = [], []
                for epoch in range(1, cfg['epochs'] + 1):
                    train_mode(model)
                    losses, norms = [], []
                    started = time.monotonic()
                    for step in range(cfg['steps_per_epoch']):
                        ids = rng.choice(train_ids, cfg['batch_size'], replace=True, p=weight)
                        flips = rng.random(cfg['batch_size']) < .5
                        all_ids.append(ids.astype(np.int32))
                        all_flips.append(flips)
                        ids_gpu = torch.as_tensor(ids, device=device)
                        flip_gpu = torch.as_tensor(flips, device=device)
                        batch = images[ids_gpu].clone()
                        batch[flip_gpu] = batch[flip_gpu].flip(-1)
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(batch, scalars[ids_gpu])[:, 0]
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets[ids_gpu])
                        assert torch.isfinite(loss), (name, fold, epoch, step)
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
                        optimizer.step()
                        losses.append(float(loss.detach()))
                        norms.append(float(norm))
                    entry = dict(model=name, fold=fold, epoch=epoch, steps=epoch * cfg['steps_per_epoch'],
                                 loss=float(np.mean(losses)), gradient_norm=float(np.mean(norms)),
                                 seconds=time.monotonic() - started)
                    history.writerow(entry)
                    writer.add_scalar('train/loss', entry['loss'], epoch)
                    writer.add_scalar('train/gradient_norm_before_clip', entry['gradient_norm'], epoch)
                    writer.flush()
                    print(json.dumps(entry), flush=True)
                assert all(torch.equal(value, bn_state(model)[key]) for key, value in original_bn.items())
                pred[name][test_ids] = predict(model, images, scalars, test_ids, device, cfg['batch_size'])
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=cfg['epochs'],
                                seed=seed, arm=name, fold=fold, config=cfg, source=source,
                                train_indices=train_ids, numpy_rng=rng.bit_generator.state,
                                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device)),
                           run / 'models' / f'{name}_{fold}_last.pt')
                np.savez_compressed(run / 'sampling' / f'{name}_{fold}.npz',
                                    train_indices=train_ids, weights=weight,
                                    sampled_indices=np.stack(all_ids), horizontal_flips=np.stack(all_flips))
                writer.close()
                writer = None
                del model, optimizer
        history_file.close()
        test_rows = [z for i, z in enumerate(rows) if residual[i]]
        predictions = {'S': reference[residual], **{name: values[residual] for name, values in pred.items()}}
        metrics = {}
        for name, values in predictions.items():
            assert np.isfinite(values).all()
            metrics[name] = score_metrics(test_rows, values, cfg['threshold'])
            metrics[name]['per_fold'] = {
                str(f): score_metrics([z for z in test_rows if int(z['fold']) == f],
                                     values[folds[residual] == f], cfg['threshold']) for f in cfg['folds']}
        g = cfg['gate']
        gates = {}
        for name in cfg['models']:
            m = metrics[name]
            gates[name] = dict(AP_gain_over_S=m['AP'] - metrics['S']['AP'] >= g['AP_gain_over_S'],
                precision=m['precision'] >= g['precision'],
                unique_positive_gt_frames=m['positive_unique_gt_frames'] >= g['unique_positive_gt_frames'],
                positive_scenes=m['positive_scenes'] >= g['positive_scenes'])
        admitted = [name for name in cfg['models'] if all(gates[name].values())]
        selected = sorted(admitted, key=lambda name: (-metrics[name]['AP'], name != 'R'))[0] if admitted else None
        save(run / 'metrics.json', dict(models=metrics, gates=gates, selected_model=selected,
                                       F_minus_R_AP=metrics['F']['AP'] - metrics['R']['AP']))
        with (run / 'oof_predictions.csv').open('w') as file:
            fields = ['sequence', 'scene', 'frame', 'index', 'owner', 'label', 'fold', 'S', 'R', 'F']
            csvwriter = csv.DictWriter(file, fieldnames=fields)
            csvwriter.writeheader()
            for i, z in enumerate(test_rows):
                csvwriter.writerow({**{k: z[k] for k in fields[:7]}, **{k: float(v[i]) for k, v in predictions.items()}})
        for path, digest in inputs.items():
            assert sha(root / path) == digest, path
        assert not git('status', '--porcelain') and git('rev-parse', 'HEAD') == source['commit']
        manifest['peak_gpu_memory_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        manifest['bn_running_stats_unchanged'] = True
        manifest['status'] = 'completed'
        rc = 0
        print(json.dumps(dict(models=metrics, gates=gates, selected_model=selected), indent=2), flush=True)
    except BaseException:
        traceback.print_exc()
        manifest['status'] = 'failed'
    finally:
        if writer is not None:
            writer.close()
        manifest['end_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        save(run / 'manifest.yaml', manifest)
        (run / 'exit_code.txt').write_text(str(rc) + '\n')
        sys.stdout.flush()
        sys.stderr.flush()
        save(run / 'output_sha256.json', {str(p.relative_to(run)): sha(p) for p in sorted(run.rglob('*'))
                                        if p.is_file() and p.name != 'output_sha256.json'})
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
