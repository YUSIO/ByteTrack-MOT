"""python -m feedback.infer: save independent causal detections and tracks."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from .model import build
from .online import OnlineFeedback
from .data import read_image, sequence_info


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--psmp-checkpoint', type=Path)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'test'], required=True)
    p.add_argument('--sequences', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--limit-frames', type=int, default=0)
    p.add_argument('--allow-untrained-feedback', action='store_true', help='Preflight only, not a result')
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if config['mode'] != 'none' and 'feedback_config' not in state and not args.allow_untrained_feedback:
        p.error('Feedback weights are absent; random modules are only permitted in explicit preflight')
    if state.get('metadata', {}).get('preflight') and not args.allow_untrained_feedback:
        p.error('Preflight detector checkpoint requires --allow-untrained-feedback')
    model = build(config, args.checkpoint).to(args.device).eval()
    predictor = None
    if config['mode'] == 'eec':
        from .psmp import load_psmp
        if args.psmp_checkpoint is None:
            p.error('--psmp-checkpoint is required for EEC')
        prior_state = torch.load(args.psmp_checkpoint, map_location='cpu', weights_only=True)
        if prior_state.get('metadata', {}).get('preflight') and not args.allow_untrained_feedback:
            p.error('Preflight PSMP checkpoint requires --allow-untrained-feedback')
        predictor = load_psmp(args.psmp_checkpoint, config['psmp']).to(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {'config': config, 'checkpoint_sha256': sha256(args.checkpoint),
               'psmp_sha256': sha256(args.psmp_checkpoint) if args.psmp_checkpoint else None,
               'split': args.split, 'gt_read': False, 'device': args.device,
               'preflight': bool(args.allow_untrained_feedback or args.limit_frames), 'sequences': []}
    for name in args.sequences:
        if Path(name).name != name:
            raise ValueError('Sequence name may not contain a path')
        path = args.data_root / args.split / name
        info = sequence_info(path)
        runner = OnlineFeedback(model, config, predictor, int(info['framerate']))
        out = args.output / name
        out.mkdir()
        length = int(info['seqlength'])
        if args.limit_frames:
            length = min(length, args.limit_frames)
        started = time.perf_counter()
        with (out/'det.txt').open('w') as det, (out/'tracks.txt').open('w') as tracks, (out/'trace.jsonl').open('w') as trace:
            for frame in range(1, length+1):
                image, ratio, shape = read_image(path, frame, info, config['input_size'])
                r = runner.step(image[None].to(args.device), shape, ratio)
                for x1,y1,x2,y2,score in r['detections']:
                    det.write(f'{frame},-1,{x1:.4f},{y1:.4f},{x2-x1:.4f},{y2-y1:.4f},{score:.6f},-1,-1,-1\n')
                for identity,x,y,w,h,score in r['tracks']:
                    tracks.write(f'{frame},{identity},{x:.1f},{y:.1f},{w:.1f},{h:.1f},{score:.6f},-1,-1,-1\n')
                trace.write(json.dumps({k:v for k,v in r.items() if k not in ('detections','tracks')})+'\n')
        summary['sequences'].append({'name': name, 'frames': length, 'seconds': time.perf_counter()-started,
                                     'det_sha256': sha256(out/'det.txt'), 'tracks_sha256': sha256(out/'tracks.txt')})
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary['sequences'][-1]), flush=True)


if __name__ == '__main__':
    main()
