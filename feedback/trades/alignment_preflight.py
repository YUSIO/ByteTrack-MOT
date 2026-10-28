"""CPU-only preflight for the identity-warp / self-history dev controls.

Checks, on two real UAVSwarm frames with the real frozen YOLOX-X and the real
Exp194 epoch10 adapter:

  * off is an exact bypass of the adapter;
  * learned runs are deterministic and leave every weight/BN tensor unchanged;
  * the identity hook receives non-zero DCN offsets and zeroing them changes
    the adapter output, while an independent zero-and-record hook reproduces
    the production hook bit-for-bit;
  * self-history (previous P3 := current P3) runs finite and differs from both
    learned and off;
  * all four arms produce parseable, non-empty detection lines on frame 2.

No training, no GT evaluation, no test access.
"""
import argparse
import contextlib
import datetime
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
import torch
from feedback.data import read_train_sequences, read_image
from .checkpoint import load_adapter, sha256
from .data import render_heat
from .model import FrozenTraDeS
from .preflight import tensor_hashes
from .alignment_counterfactual import detection_lines


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(2)
    if sha256(args.checkpoint) != 'e6a7c579fa78e19d331312a2a2324ca594f666112c9f20219f30621db4c3fc21':
        raise ValueError('Expected immutable Exp141 epoch80 detector')
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Commit implementation before preflight')

    model = FrozenTraDeS(args.checkpoint).eval()
    metadata = load_adapter(model, args.adapter, args.checkpoint)
    model.detector.head.decode_in_inference = False
    before = tensor_hashes(model)

    name = 'UAVSwarm-01'
    data = read_train_sequences(args.data_root, [name])
    path, info, frames = data[name]
    evidence = dict(phase='alignment_counterfactual_preflight', status='running',
                    device='cpu', threads=2, sequence=name, frames=[1, 2],
                    test_access=False, optimizer_steps=0,
                    code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    code_dirty=False, adapter_metadata=metadata,
                    base_sha256=sha256(args.checkpoint), adapter_sha256=sha256(args.adapter),
                    python=sys.version, torch=str(torch.__version__),
                    platform=platform.platform(),
                    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(args.output / 'evidence.json', evidence)

    # Previous-frame GT heat is used only to exercise the active adapter path.
    image1, ratio1, shape1 = read_image(path, 1, info, model.input_size)
    image2, ratio2, shape2 = read_image(path, 2, info, model.input_size)
    if (ratio1, shape1) != (ratio2, shape2):
        raise ValueError('Geometry drift across the two preflight frames')

    def gt_heat(frame):
        boxes = np.asarray(list(frames[frame].values()), np.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        return render_heat(boxes * ratio1, model.input_size)[None]

    features1 = model.detector.backbone(image1[None])
    raw1 = features1[0]
    heat_empty = torch.zeros((1, 1, *raw1.shape[2:]), device='cpu')
    # The empty-heat bypass is exercised only to prove it is an exact bypass;
    # every real diagnostic below uses the previous-frame GT heat.
    bypass, cva_bypass, residual_bypass = model.adapter(raw1, raw1, heat_empty, enabled=True)
    if not torch.equal(bypass, raw1) or cva_bypass is not None or float(residual_bypass.abs().max()) != 0.0:
        raise RuntimeError('Empty history is not an exact bypass')

    features = model.detector.backbone(image2[None])
    raw = features[0]
    previous = raw1
    heat = gt_heat(1)
    target_ratio = ratio2
    head = model.detector.head

    def run_arm(label, previous_input, enabled=True, hook_factory=None):
        manager = contextlib.nullcontext() if hook_factory is None else hook_factory()
        with manager:
            refined, cva, residual = model.adapter(raw, previous_input, heat, enabled=enabled)
        level = (refined, *features[1:])
        prediction = head(level)
        lines = detection_lines(prediction, head, target_ratio, 2)
        return dict(label=label, refined=refined, cva=cva, residual=residual,
                    lines=lines, detections=len([x for x in lines.splitlines() if x.strip()]))

    learned = run_arm('learned', previous)
    learned_repeat = run_arm('learned_repeat', previous)
    if not torch.equal(learned['refined'], learned_repeat['refined']):
        raise RuntimeError('Learned arm is not deterministic')

    off = run_arm('off', previous, enabled=False)
    if not torch.equal(off['refined'], raw) or off['cva'] is not None:
        raise RuntimeError('Off is not an exact adapter bypass')

    seen = []

    @contextlib.contextmanager
    def zero_and_record():
        def hook(module, hook_args):
            support, offset, mask = hook_args
            seen.append(float(offset.abs().max()))
            return support, offset * 0, mask
        handle = model.adapter.core.dcn1_1.register_forward_pre_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    manual = run_arm('identity_manual', previous, hook_factory=zero_and_record)
    from .alignment_counterfactual import identity_offsets
    identity = run_arm('identity', previous, hook_factory=lambda: identity_offsets(model.adapter.core, True))
    if not seen or max(seen) <= 0.0:
        raise RuntimeError('DCN offsets were not populated before the identity hook')
    if not torch.equal(manual['refined'], identity['refined']):
        raise RuntimeError('Production identity hook differs from manual zeroing')
    if torch.equal(identity['refined'], learned['refined']):
        raise RuntimeError('Identity arm did not change the adapter output')

    self_arm = run_arm('self', raw)
    if torch.equal(self_arm['refined'], learned['refined']) or torch.equal(self_arm['refined'], raw):
        raise RuntimeError('Self-history arm did not change the adapter output')

    after = tensor_hashes(model)
    if after != before:
        raise RuntimeError('Preflight changed weights or BN state')
    for arm in (learned, off, identity, manual, self_arm):
        if arm['detections'] <= 0:
            raise RuntimeError('Empty detection lines for ' + arm['label'])
        for line in arm['lines'].splitlines():
            row = list(map(float, line.split(',')))
            if len(row) != 10 or not np.isfinite(row).all():
                raise RuntimeError('Malformed detection line for ' + arm['label'])

    evidence.update(status='passed',
                    dcn_offset_max=max(seen),
                    residual_rms={k: float(v['residual'].float().square().mean().sqrt())
                                  for k, v in dict(learned=learned, identity=identity,
                                                   self=self_arm, off=off).items()},
                    detections={k: v['detections'] for k, v in
                                dict(learned=learned, identity=identity, self=self_arm, off=off).items()},
                    identity_equals_manual_zeroing=True,
                    identity_differs_from_learned=not torch.equal(identity['refined'], learned['refined']),
                    self_differs_from_learned=not torch.equal(self_arm['refined'], learned['refined']),
                    empty_history_exact_bypass=True,
                    off_exact_bypass=True,
                    weights_and_bn_unchanged=True,
                    finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(args.output / 'evidence.json', evidence)
    print(json.dumps({k: evidence[k] for k in ('status', 'dcn_offset_max', 'residual_rms',
                                              'detections', 'identity_equals_manual_zeroing')}, indent=2))


if __name__ == '__main__':
    main()
