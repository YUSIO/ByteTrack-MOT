"""Bounded CPU audit: frozen real detector, adapter updates, exact OFF recovery.

No formal trainer or test-set path. Diagnostic adapter is never an initializer.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import traceback

import numpy as np
import torch

from .data import read_image, sequence_info, read_train_sequences
from .weak import WeakDetector, WeakOnline, heat_from_record, postprocess_rows, DetectionTracker
from .weak_check import formatted_tracks
from .weak_history import sha256, make_optimizer, train_batch, assert_frozen_detector, load_snapshot


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def state_hashes(model):
    return {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, value in model.state_dict().items()}


def check(root, protocol, output, manifest):
    expected = protocol['initialization']['checkpoint_sha256']
    checkpoint = root / protocol['initialization']['checkpoint']
    assert sha256(checkpoint) == expected
    manifest['input_hashes'][str(checkpoint)] = expected
    data = root / protocol['dataset']['root']
    names = protocol['preflight']['sequences']
    assert set(names) <= set(protocol['dataset']['fit_sequences'])
    assert protocol['training']['trainable'] == 'adapter_only'
    assert set(protocol['arms']) == {'B0', 'T', 'TU'}
    torch.manual_seed(0)
    model = WeakDetector(checkpoint).eval()
    model.detector.head.decode_in_inference = False
    before = state_hashes(model.detector)
    buffers_before = {k: v.clone() for k, v in model.detector.named_buffers()}
    adapter_before = state_hashes(model.adapter)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 5264
    runner = WeakOnline(model, 'TU', expected)
    samples = []
    for name in names:
        path = data / 'train' / name
        info = sequence_info(path)
        manifest['input_hashes'][str(path / 'seqinfo.ini')] = sha256(path / 'seqinfo.ini')
        runner.reset(name)
        for frame in range(1, 5):
            image_path = path / info.get('imdir', 'img1') / f"{frame:06d}{info.get('imext', '.jpg')}"
            manifest['input_hashes'][str(image_path)] = sha256(image_path)
            image, ratio, shape = read_image(path, frame, info, (800, 1440))
            image = image[None]
            history = runner.state.snapshot(name, frame, expected)
            heat = heat_from_record(history, ratio, (100, 180))[None]
            with torch.no_grad():
                raw = model.detector(image).clone()
                assert torch.equal(raw, model(image, heat))
                assert torch.equal(raw, model(image, heat, enabled=False))
            dets = postprocess_rows(raw, model.detector.head, ratio)
            tracks, audit = runner.tracker.update(dets, shape)
            runner.state.advance(name, frame, dets, audit)
            samples.append(dict(name=name, frame=frame, image=image, heat=heat,
                                ratio=ratio, shape=shape, raw=raw, dets=dets,
                                tracks=formatted_tracks(frame, tracks)))
            print(f'identity_before {name} frame={frame}', flush=True)
    # Only current-frame supervision reads GT, after all histories are fixed.
    annotations = read_train_sequences(data, names)
    for name in names:
        path = data / 'train' / name / 'gt/gt.txt'
        manifest['input_hashes'][str(path)] = sha256(path)
    batches = []
    for name in names:
        part = [s for s in samples if s['name'] == name and s['frame'] in (3, 4)]
        targets = []
        for s in part:
            boxes = np.asarray(list(annotations[name][2][s['frame']].values()), np.float32).reshape(-1, 4)
            t = torch.zeros(len(boxes), 5)
            t[:, 1:3] = torch.from_numpy((boxes[:, :2] + boxes[:, 2:] / 2) * s['ratio'])
            t[:, 3:] = torch.from_numpy(boxes[:, 2:] * s['ratio'])
            targets.append(t)
        padded = torch.zeros(2, max(1, max(len(t) for t in targets)), 5)
        for i, t in enumerate(targets):
            padded[i, :len(t)] = t
        batches.append((torch.cat([s['image'] for s in part]),
                        torch.cat([s['heat'] for s in part]), padded))
    optimizer = make_optimizer(model)
    losses, grads = [], []
    for step in range(4):
        losses.append(train_batch(model, optimizer, *batches[step % len(batches)]))
        grads.append([p.grad.norm().item() for p in model.adapter.parameters()])
        assert_frozen_detector(model)
        assert before == state_hashes(model.detector), 'Detector state changed'
        assert all(torch.equal(v, buffers_before[k]) for k, v in model.detector.named_buffers())
        print(json.dumps({'probe_step': step + 1, 'loss': losses[-1], 'grad_norms': grads[-1]}), flush=True)
    assert all(grads[i][1] > 0 for i in range(4))
    assert any(g[0] > 0 for g in grads[1:])
    after_adapter = state_hashes(model.adapter)
    assert all(after_adapter[k] != v for k, v in adapter_before.items())
    model.eval()
    changed = 0
    for name in names:
        tracker_off, tracker_zero = DetectionTracker(), DetectionTracker()
        for s in [s for s in samples if s['name'] == name]:
            with torch.no_grad():
                off = model(s['image'], s['heat'], enabled=False)
                zero = model(s['image'], torch.zeros_like(s['heat']))
                on = model(s['image'], s['heat'])
            assert torch.equal(off, s['raw']) and torch.equal(zero, s['raw'])
            changed += int(not torch.equal(on, s['raw']))
            for raw, tracker in ((off, tracker_off), (zero, tracker_zero)):
                dets = postprocess_rows(raw, model.detector.head, s['ratio'])
                np.testing.assert_array_equal(dets, s['dets'])
                tracks, _ = tracker.update(dets, s['shape'])
                assert formatted_tracks(s['frame'], tracks) == s['tracks']
            print(f'identity_after {name} frame={s["frame"]}', flush=True)
    assert changed > 0
    assert before == state_hashes(model.detector)
    snapshot = output / 'diagnostic_adapter.pt'
    torch.save({'weak_config': 'exp162_frozen_v3', 'preflight': True,
                'base_checkpoint': str(checkpoint.resolve()), 'base_checkpoint_sha256': expected,
                'adapter': model.adapter.state_dict()}, snapshot)
    del optimizer, model
    reloaded = load_snapshot(snapshot).eval()
    reloaded.detector.head.decode_in_inference = False
    assert state_hashes(reloaded.detector) == before
    assert state_hashes(reloaded.adapter) == after_adapter
    with torch.no_grad():
        assert torch.equal(reloaded(s['image'], s['heat']), on)
        assert torch.equal(reloaded(s['image'], s['heat'], enabled=False), s['raw'])
    for path, digest in manifest['input_hashes'].items():
        assert sha256(path) == digest
    dump(output / 'detector_tensor_hashes.json', before)
    return {'passed': True, 'identity_frames': len(samples), 'optimizer_steps': 4,
            'trainable_parameters': 5264, 'detector_tensor_count': len(before),
            'detector_all_parameters_and_buffers_bitwise_unchanged': True,
            'BN_buffers_bitwise_unchanged': True, 'BN_all_eval': True,
            'detector_gradients_absent': True, 'off_raw_detections_tracks_equal_initial': True,
            'zero_heat_raw_detections_tracks_equal_initial': True,
            'nonzero_feedback_changed_raw_frames': changed,
            'losses': losses, 'adapter_gradient_norms': grads,
            'both_adapter_layers_changed': True, 'reload_on_and_off_identical': True,
            'initial_checkpoint_sha256': expected, 'adapter_checkpoint_sha256': sha256(snapshot),
            'formal_feedback_training_started': False, 'test_read': False,
            'diagnostic_adapter_reusable_for_formal_training': False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    manifest = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'status': 'running', 'phase': 'bounded_frozen_adapter_preflight',
                'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'code_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True)),
                'python': sys.version, 'python_executable': sys.executable, 'torch': str(torch.__version__),
                'host': platform.node(), 'device': 'cpu', 'threads': 4, 'seed': 0,
                'protocol_sha256': sha256(args.protocol), 'input_hashes': {}}
    dump(args.output / 'manifest.yaml', manifest)
    (args.output / 'command.txt').write_text(shlex.join([sys.executable, '-m', 'feedback.weak_frozen_check', *sys.argv[1:]]) + '\n')
    rc = 1
    try:
        assert not manifest['code_dirty'], 'Commit source before real preflight'
        protocol = json.loads(args.protocol.read_text())
        dump(args.output / 'frozen_protocol.json', protocol)
        dump(args.output / 'metrics.json', check(args.root, protocol, args.output, manifest))
        manifest['status'] = 'completed'
        rc = 0
    except Exception:
        manifest['status'] = 'failed'
        (args.output / 'exception.txt').write_text(traceback.format_exc())
        raise
    finally:
        manifest['exit_code'] = rc
        manifest['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        dump(args.output / 'manifest.yaml', manifest)
        (args.output / 'exit_code.txt').write_text(str(rc) + '\n')


if __name__ == '__main__':
    main()
