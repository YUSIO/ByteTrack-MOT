"""CPU-only input/configuration/COCO-load preflight. No forward or optimizer.

This command cannot launch training; tools/train.py remains a separate action.
"""
import argparse
import itertools
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from exps.example.uavswarm.yolox_x_exp162_b0_fit28 import (
    COCO_SHA256, Exp, FIT_SEQUENCES, file_sha256,
)
from yolox.utils import load_ckpt


def reject(call):
    try:
        call()
    except (ValueError, FileNotFoundError):
        return True
    raise AssertionError('Expected unsafe configuration to be rejected')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-view', type=Path, required=True)
    p.add_argument('--dev-view', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Do not overwrite a preflight attempt')
    args.output.mkdir(parents=True)
    os.environ['UAVSWARM_V1_ROOT'] = str(args.data_view.resolve())
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.set_num_threads(4)
    # Only deserialize the known COCO bytes; never accept a UAVSwarm-trained X.
    checksum = file_sha256(args.checkpoint)
    if checksum != COCO_SHA256:
        raise ValueError('Initialization must be the pinned official COCO YOLOX-X')
    exp = Exp()
    effective = {k: getattr(exp, k) for k in (
        'depth', 'width', 'num_classes', 'seed', 'max_epoch', 'train_ann', 'val_ann',
        'input_size', 'test_size', 'random_size', 'basic_lr_per_img', 'warmup_epochs',
        'no_aug_epochs', 'eval_interval', 'data_num_workers', 'ema', 'scheduler',
        'momentum', 'weight_decay', 'degrees', 'translate', 'scale', 'mscale',
        'shear', 'perspective', 'enable_mixup', 'test_conf', 'nmsthre')}
    assert exp.max_epoch == 80 and exp.num_classes == 1 and exp.seed == 0
    assert exp.basic_lr_per_img * 4 == .0000625
    guards = {'override_train_ann': reject(lambda: exp.merge(['train_ann', 'train.json'])),
              'testdev_loader': reject(lambda: exp.get_eval_loader(1, False, True)),
              'testdev_evaluator': reject(lambda: exp.get_evaluator(1, False, True)),
              'wrong_batch': reject(lambda: exp.get_data_loader(2, False))}
    os.environ['UAVSWARM_V1_ROOT'] = str(args.dev_view.resolve())
    guards['dev_as_fit'] = reject(Exp)
    os.environ['UAVSWARM_V1_ROOT'] = str(args.data_view.resolve())
    exp.data_num_workers = 0  # Audit only: no worker processes or GPU allocation.
    loader = exp.get_data_loader(4, False)
    dataset = loader.dataset._dataset
    assert len(dataset) == 4653 and len(loader) == 1164
    seqs = {im['file_name'].split('/')[0] for im in dataset.coco.imgs.values()}
    assert seqs == set(FIT_SEQUENCES)
    assert len(dataset.coco.anns) == 36030
    batches = list(itertools.islice(iter(loader.batch_sampler), len(loader)))
    indices = [int(v[1]) for batch in batches for v in batch]
    assert len(indices) == 4656 and len(set(indices)) == 4653
    assert all(len(batch) == 4 for batch in batches)
    eval_loader = exp.get_eval_loader(1, False)
    assert eval_loader.dataset.ids == dataset.ids
    # Verify actual train image paths through the production loader, without model forward.
    samples = []
    for index in (0, len(dataset) - 1):
        image, target, info, identity = dataset.pull_item(index)
        assert image.shape[:2] == tuple(info[:2]) and target.shape[1] == 6
        samples.append({'image_id': int(identity[0]), 'file': info[-1], 'targets': len(target)})
    model = exp.get_model().cpu()
    state = model.state_dict()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)['model']
    assert all(k in ckpt for k in state)
    mismatches = [k for k, v in state.items() if v.shape != ckpt[k].shape]
    expected = {f'head.cls_preds.{i}.{suffix}' for i in range(3) for suffix in ('weight', 'bias')}
    assert set(mismatches) == expected, mismatches
    before = {k: state[k].clone() for k in mismatches}
    model = load_ckpt(model, ckpt)
    after = model.state_dict()
    assert all(torch.equal(after[k], ckpt[k]) for k in state if k not in expected)
    assert all(torch.equal(after[k], before[k]) for k in expected)
    assert all(ckpt[k].shape[0] == 80 and after[k].shape[0] == 1 for k in expected)
    code_root = Path(__file__).resolve().parents[1]
    git = lambda *a: subprocess.check_output(['git', '-C', str(code_root), *a], text=True).strip()
    result = {'status': 'passed', 'phase': 'local_CPU_configuration_and_initialization_only',
        'training_started': False, 'optimizer_created': False, 'model_forward_executed': False,
        'test_read': False, 'checkpoint_sha256': checksum, 'checkpoint_bytes': args.checkpoint.stat().st_size,
        'checkpoint_tensor_count': len(ckpt), 'loaded_matching_tensors': len(state) - len(mismatches),
        'classification_tensors_retained_random': sorted(mismatches),
        'unexpected_checkpoint_keys': sorted(set(ckpt) - set(state)),
        'initialization_policy': 'COCO_matching_shapes_only_6_single_class_outputs_seed0_initialized',
        'effective_configuration': effective, 'batch_size': 4,
        'steps_per_epoch': len(loader), 'batches_are_full': True, 'sample_slots_per_epoch': len(indices),
        'sampling': 'InfiniteSampler_continuous_stream_not_last_batch_size1',
        'fit_annotation_sha256': file_sha256(args.data_view / 'annotations/fit28.json'),
        'fit_sequences': sorted(seqs), 'monitoring': 'fit28_only_not_checkpoint_selection',
        'guards': guards, 'loader_samples': samples,
        'python_executable': sys.executable, 'python': platform.python_version(),
        'torch': torch.__version__, 'numpy': np.__version__, 'device_used': 'cpu',
        'cuda_available': torch.cuda.is_available(), 'remote_CUDA_runtime_gate': 'pending_future_execution',
        'code_commit': git('rev-parse', 'HEAD'), 'code_dirty': bool(git('status', '--porcelain')),
        'source_hashes': {str(q.relative_to(code_root)): file_sha256(q) for q in
                          (Path(__file__), code_root / 'exps/example/uavswarm/yolox_x_exp162_b0_fit28.py',
                           code_root / 'exps/example/uavswarm/yolox_x_uavswarmv1_full.py',
                           code_root / 'yolox/utils/checkpoint.py')}}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
