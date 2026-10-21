"""Replay frozen individual gate changes, retaining the original future policy."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

import motmetrics as mm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fctrack.action_diagnostic import DecisionController
from fctrack.run import run_arm
from homa.evaluate import evaluate

mm.lap.default_solver = 'lap'
FIELDS = {'FP': 'num_false_positives', 'FN': 'num_misses', 'IDSW': 'num_switches'}


def counts(record):
    return {k: int(record['motmetrics'][v]) for k, v in FIELDS.items()}


def prefix_counts(gt, prediction, last_frame):
    g = gt[gt.index.get_level_values(0) <= last_frame]
    p = prediction[prediction.index.get_level_values(0) <= last_frame]
    acc = mm.utils.compare_to_groundtruth(g, p, 'iou', distth=.5)
    row = mm.metrics.create().compute(acc, metrics=list(FIELDS.values())).iloc[0]
    return {k: int(row[v]) for k, v in FIELDS.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--targets', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True, help='immutable previous run directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit', type=int, default=0, help='preflight only')
    a = p.parse_args()
    root, out = a.root.resolve(), a.output.resolve()
    cfg = json.loads(a.config.read_text())
    targets = json.loads(a.targets.read_text())['targets']
    if a.limit:
        targets = targets[:a.limit]
    baseline = json.loads((a.baseline / 'output/metrics/A_orig.json').read_text())
    cell = cfg['cells']['v1_dev_last']
    out.mkdir(parents=True, exist_ok=False)
    summaries = []
    for i, target in enumerate(targets):
        case = out / 'cases' / f'event_{i:03d}'
        case.mkdir(parents=True)
        name, frame = target['sequence'], target['frame']
        one_cell = {**cell, 'sequences': [name]}
        spec = {'mode': target['mode'], 'label': 'any', 'kinds': [target['kind']], 'target': target}
        arm = copy.deepcopy(cfg['arms']['A_orig'])
        arm['diagnostic'] = spec
        factory = lambda n, fc, data: DecisionController(n, fc, data, spec, case / 'events')
        names, stats, digests = run_arm(root, cfg, one_cell, arm, case / 'tracks', a.device, 0,
                                       controller_factory=factory)
        assert names == [name], names
        prediction = case / 'tracks' / f'{name}.txt'
        reference = a.baseline / 'output/tracks/A_orig' / f'{name}.txt'
        before = lambda path: [line for line in path.read_bytes().splitlines() if int(line.split(b',', 1)[0]) < frame]
        if before(reference) != before(prediction):
            raise RuntimeError(f'Prefix changed before target: {target}')
        events = [json.loads(x) for x in (case / 'events' / f'{name}.jsonl').read_text().splitlines()]
        changed = [e for e in events if e['decision_changed']]
        if len(changed) != 1 or changed[0]['signature'] != target['signature']:
            raise RuntimeError('Single-event intervention did not preserve the frozen query')
        metric_path = case / 'canonical_metrics.json'
        e = evaluate(root / cell['dataset_root'], cell['split'], case / 'tracks', metric_path, names)
        e['protocol'].update(gt_used_only_for_evaluation=False, gt_assisted_single_event_diagnostic=True)
        metric_path.write_text(json.dumps(e, indent=2, allow_nan=False) + '\n')
        current, original = counts(e['overall']), counts(baseline['per_sequence'][name])
        delta = {k: current[k] - original[k] for k in FIELDS}
        gt_path = root / cell['dataset_root'] / cell['split'] / name / 'gt/gt.txt'
        gt = mm.io.loadtxt(str(gt_path), fmt='mot15-2D', min_confidence=1)
        pred = mm.io.loadtxt(str(prediction), fmt='mot15-2D', min_confidence=-1)
        ref = mm.io.loadtxt(str(reference), fmt='mot15-2D', min_confidence=-1)
        pc, pb = prefix_counts(gt, pred, frame), prefix_counts(gt, ref, frame)
        immediate = {k: pc[k] - pb[k] for k in FIELDS}
        future = {k: delta[k] - immediate[k] for k in FIELDS}
        # Check the prefix helper against canonical full CLEAR counts before attributing future effects.
        last = max(int(gt.index.get_level_values(0).max()), int(pred.index.get_level_values(0).max()) if len(pred) else 0,
                   int(ref.index.get_level_values(0).max()) if len(ref) else 0)
        if prefix_counts(gt, pred, last) != current or prefix_counts(gt, ref, last) != original:
            raise RuntimeError('Prefix metric helper differs from canonical evaluator')
        summary = {'event': i, 'target': target, 'actual_outcome': changed[0]['outcome'],
                   'prefix_identical': True, 'changed_decisions': 1, 'delta_full_errors': delta,
                   'delta_through_action_frame_errors': immediate, 'delta_future_errors': future,
                   'net_errors_saved': -sum(delta.values()), 'future_errors_saved': -sum(future.values()),
                   'all_tracks_identical': reference.read_bytes() == prediction.read_bytes(),
                   'sequence_metrics': e['overall'], 'stats': stats, 'track_sha256': digests[name]}
        (case / 'event_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        summaries.append({k: v for k, v in summary.items() if k != 'sequence_metrics'})
        print(i, name, target['kind'], target['mode'], frame, 'saved', summary['net_errors_saved'],
              'future_saved', summary['future_errors_saved'], flush=True)
    result = {'scope': 'independent one-event replays; effects are not additive',
              'target_file_sha256': hashlib.sha256(a.targets.read_bytes()).hexdigest(),
              'event_count': len(summaries), 'events': summaries}
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (out / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['event', 'sequence', 'frame', 'kind', 'mode', 'actual_outcome',
                                             'net_errors_saved', 'future_errors_saved', 'all_tracks_identical'])
        writer.writeheader()
        for event in summaries:
            writer.writerow({**{k: event[k] for k in writer.fieldnames if k in event},
                             **{k: event['target'][k] for k in ('sequence', 'frame', 'kind', 'mode')}})


if __name__ == '__main__':
    main()
