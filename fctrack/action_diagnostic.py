"""Exp139: preserve scores/boxes/order and intervene only at verifier decision gates.

This is a ground-truth-assisted diagnostic, never a deployable tracker. One-event
replays rerun the same prefix and change a single decision; all later decisions
use the original policy in the changed state. Event effects must not be summed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fctrack.run import run_arm, KEYS
from fctrack.tracker import iou_matrix
from homa.evaluate import evaluate


def signature(event):
    fields = {k: event[k] for k in ('frame', 'candidate_index', 'proposal', 'final', 'p', 'native_accept')}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


class DecisionController:
    def __init__(self, name, fc, seq_dir, spec, destination):
        self.name, self.fc, self.spec, self.destination = name, fc, spec, destination
        raw = np.loadtxt(seq_dir / 'gt/gt.txt', delimiter=',', ndmin=2)
        raw = raw[raw[:, 6] >= 1]
        self.gt = {int(f): raw[raw[:, 0] == f] for f in np.unique(raw[:, 0])}
        self.owner, self.events, self.changed = {}, [], 0
        fc.decision_hook = self.decide

    def decide(self, kind, event):
        event['kind'], event['sequence'] = kind, self.name
        event['signature'] = signature(event)
        gt = self.gt.get(event['frame'], np.empty((0, 9)))
        overlaps = iou_matrix([event['final']], gt[:, 2:6])[0] if len(gt) else np.empty(0)
        event['max_iou'] = float(overlaps.max()) if len(overlaps) else 0.0
        event['hit_gt_ids'] = [int(x) for x in gt[overlaps >= 0.5, 1]]
        event['owner_gt_id'] = self.owner.get(event.get('track_id'))
        event['own_hit'] = event['owner_gt_id'] in event['hit_gt_ids'] if event['owner_gt_id'] is not None else None
        event['any_hit'] = bool(event['hit_gt_ids'])
        # Ground truth owner is for diagnostics only; absent owners are not silently labelled negative.
        hit = event['own_hit'] if self.spec.get('label') == 'own' and kind == 'fill' else event['any_hit']
        native = bool(event['native_accept'])
        chosen = native
        if kind in self.spec.get('kinds', ['fill', 'birth']):
            target = self.spec.get('target')
            selected = target is None or all(event.get(k) == target[k] for k in ('sequence', 'kind', 'frame', 'candidate_index'))
            if selected and target is not None and event['signature'] != target['signature']:
                raise RuntimeError('Single-event prefix/query differs from frozen target')
            if selected and hit is not None:
                mode = self.spec['mode']
                if mode == 'veto':
                    chosen = native and hit
                elif mode == 'admit':
                    chosen = native or hit
                elif mode == 'oracle':
                    chosen = bool(hit)
                elif mode != 'noop':
                    raise ValueError(mode)
        event['override_accept'] = bool(chosen)
        event['decision_changed'] = bool(chosen != native)
        self.changed += int(event['decision_changed'])
        self.events.append(event)
        return chosen

    def observe(self, frame, rows):
        # Attribute historical ownership only after a real detector update, with one-to-one IoU50.
        gt = self.gt.get(frame, np.empty((0, 9)))
        real = [(tid, b) for tid, b, _ in rows if self.fc.meta.get(tid, {}).get('real') == frame]
        if len(gt) and real:
            overlaps = iou_matrix([b for _, b in real], gt[:, 2:6])
            ii, jj = linear_sum_assignment(-(overlaps * (overlaps >= 0.5)))
            for i, j in zip(ii, jj):
                if overlaps[i, j] >= 0.5:
                    self.owner[real[i][0]] = int(gt[j, 1])
        for event in reversed(self.events):
            if event['frame'] != frame:
                break
            if event.get('outcome') == 'written':
                matches = [(tid, b) for tid, b, _ in rows if np.array_equal(np.asarray(b), np.asarray(event['final']))]
                event['output_ids_after_relabel'] = [int(tid) for tid, _ in matches]

    def finish(self):
        target = self.spec.get('target')
        if target is not None and target['sequence'] == self.name and self.changed != 1:
            raise RuntimeError(f'Single-event replay must change exactly one decision, got {self.changed}')
        self.destination.mkdir(parents=True, exist_ok=True)
        (self.destination / f'{self.name}.jsonl').write_text(''.join(json.dumps(e, allow_nan=False) + '\n' for e in self.events))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--cell', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--arms', nargs='+')
    a = p.parse_args()
    root, out = a.root.resolve(), a.output.resolve()
    cfg = json.loads(a.config.read_text())
    cell = cfg['cells'][a.cell]
    (out / 'metrics').mkdir(parents=True, exist_ok=False)
    results = {}
    for name in a.arms or cfg['arms']:
        arm = cfg['arms'][name]
        diagnostic = arm.get('diagnostic')
        factory = None
        if diagnostic is not None:
            factory = lambda seq, fc, data, d=diagnostic, n=name: DecisionController(seq, fc, data, d, out / 'events' / n)
        tracks = out / 'tracks' / name
        names, stats, hashes = run_arm(root, cfg, cell, arm, tracks, a.device, a.limit,
                                       out / 'attempts' / name, controller_factory=factory)
        metric_path = out / 'metrics' / f'{name}.json'
        e = evaluate(root / cell['dataset_root'], cell['split'], tracks, metric_path, names)
        assisted = diagnostic is not None and diagnostic['mode'] != 'noop'
        e['protocol']['gt_used_only_for_evaluation'] = not assisted
        e['protocol']['gt_assisted_decision_diagnostic'] = assisted
        e['protocol']['gt_used_for_posthoc_event_labels'] = diagnostic is not None
        metric_path.write_text(json.dumps(e, indent=2, allow_nan=False) + '\n')
        m, h = e['overall']['motmetrics'], e['overall']['hota']
        overall = dict(MOTA=m['mota'], IDF1=m['idf1'], HOTA=h['HOTA'], DetA=h['DetA'], AssA=h['AssA'],
                       FP=m['num_false_positives'], FN=m['num_misses'], IDSW=m['num_switches'], GT=m['num_objects'])
        results[name] = dict(overall=overall, stats=stats, track_sha256=hashes,
                             per_sequence=e['per_sequence'], diagnostic=diagnostic)
        if cell.get('accepted_reference_tracks'):
            reference = root / cell['accepted_reference_tracks']
            results[name]['identical_to_accepted_reference'] = [sum((reference / f'{s}.txt').read_bytes() == (tracks / f'{s}.txt').read_bytes() for s in names), len(names)]
        if 'A_orig' in results:
            results[name]['vs_A_orig'] = {k: overall[k] - results['A_orig']['overall'][k] for k in KEYS}
        (out / 'metrics.json').write_text(json.dumps(dict(cell=a.cell, arms=results), indent=2, allow_nan=False) + '\n')
        print(name, json.dumps(overall), flush=True)
        if name == 'A_orig':
            if 'A_uninstrumented' in results and results['A_uninstrumented']['track_sha256'] != hashes:
                raise RuntimeError('No-op hook changed the baseline tracks; stop before interventions')
            if 'identical_to_accepted_reference' in results[name]:
                same, total = results[name]['identical_to_accepted_reference']
                if same != total:
                    raise RuntimeError('A_orig differs from immutable reference; stop before interventions')
    if 'A_uninstrumented' in results and 'A_orig' in results:
        if results['A_uninstrumented']['track_sha256'] != results['A_orig']['track_sha256']:
            raise RuntimeError('No-op hook changed the baseline tracks')
    if 'A_orig' in results and 'identical_to_accepted_reference' in results['A_orig']:
        n, total = results['A_orig']['identical_to_accepted_reference']
        if n != total:
            raise RuntimeError('A_orig differs from the accepted immutable reference')


if __name__ == '__main__':
    main()
