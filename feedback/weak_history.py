"""Immutable epoch-snapshot history export and current-GT-only training data.

There is deliberately no multi-epoch training CLI in this preflight delivery.
Every call to export_snapshot reloads a specific snapshot and resets each sequence.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import read_image, sequence_info, read_train_sequences
from .weak import WeakDetector, WeakOnline, heat_from_record


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_snapshot(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('weak_config') == 'exp162_v1':
        model = WeakDetector()
        model.load_state_dict(state['model'], strict=True)
        return model
    del state
    return WeakDetector(path)


def export_snapshot(snapshot, root, sequences, output, arm, device='cpu', max_frames=None):
    """Image-only rollout. Does not call any annotation reader."""
    root, output = Path(root), Path(output)
    if not sequences or len(sequences) != len(set(sequences)):
        raise ValueError('Explicit unique fit sequence list required')
    for name in sequences:
        if Path(name).name != name or not int(name.split('-')[-1]) % 2:
            raise ValueError('Only official train sequence directories are allowed')
    output.mkdir(parents=True, exist_ok=False)
    checksum = sha256(snapshot)
    model = load_snapshot(snapshot).to(device).eval()
    runner = WeakOnline(model, arm, checksum)
    manifest = {'snapshot_sha256': checksum, 'arm': arm, 'files': {}, 'frames': 0,
                'gt_access': False, 'sequences': sequences, 'input_files': {}}
    for name in sequences:
        path = root/'train'/name
        info = sequence_info(path)
        runner.reset(name)
        history_path = output/(name+'.jsonl')
        manifest['input_files'][str(path/'seqinfo.ini')] = sha256(path/'seqinfo.ini')
        length = int(info['seqlength'])
        if max_frames is not None:
            length = min(length, max_frames)
        with history_path.open('w') as stream:
            for frame in range(1, length+1):
                image_path = path/info.get('imdir','img1')/f"{frame:06d}{info.get('imext','.jpg')}"
                manifest['input_files'][str(image_path)] = sha256(image_path)
                image, ratio, shape = read_image(path, frame, info, (800,1440))
                value = runner.step(image[None].to(device), name, frame, shape, ratio)
                stream.write(json.dumps(value['history'])+'\n')
                manifest['frames'] += 1
        manifest['files'][name] = sha256(history_path)
    if sha256(snapshot) != checksum:
        raise ValueError('Snapshot changed during export')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


class SnapshotPairs(Dataset):
    def __init__(self, root, sequences, cache, expected_snapshot, expected_arm):
        cache = Path(cache)
        manifest = json.loads((cache/'manifest.json').read_text())
        if manifest['snapshot_sha256'] != expected_snapshot or manifest['arm'] != expected_arm:
            raise ValueError('Wrong epoch snapshot or arm')
        if manifest['sequences'] != sequences or manifest['gt_access']:
            raise ValueError('History partition or GT provenance mismatch')
        self.data = read_train_sequences(root, sequences)
        self.index = []
        for name in sequences:
            file = cache/(name+'.jsonl')
            if sha256(file) != manifest['files'][name]:
                raise ValueError('History file changed')
            records = [json.loads(line) for line in file.read_text().splitlines()]
            if len(records) != int(self.data[name][1]['seqlength']):
                raise ValueError('Epoch history must cover all sequence frames')
            for frame, record in enumerate(records, 1):
                if record['arm'] != expected_arm:
                    raise ValueError('History row arm mismatch')
                heat_from_record(record, 1., (100,180), name, frame, expected_snapshot)
                self.index.append((name, frame, record))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        name, frame, history = self.index[index]
        path, info, frames = self.data[name]
        image, ratio, _ = read_image(path, frame, info, (800,1440))
        boxes = np.asarray(list(frames[frame].values()), np.float32).reshape(-1,4)
        targets = np.zeros((len(boxes),5), np.float32)
        targets[:,1:3] = (boxes[:,:2]+boxes[:,2:]/2)*ratio
        targets[:,3:] = boxes[:,2:]*ratio
        heat = heat_from_record(history, ratio, (100,180), name, frame)
        return image, heat.detach(), torch.from_numpy(targets), (name,frame)


def collate(batch):
    images, heats, targets, identities = zip(*batch)
    padded = torch.zeros((len(batch), max(1,max(len(t) for t in targets)),5))
    for i, target in enumerate(targets):
        padded[i,:len(target)] = target
    return torch.stack(images), torch.stack(heats), padded, identities


def train_batch(model, optimizer, images, heat, targets):
    """One bounded optimizer step; caller owns budget and immutable run record."""
    model.train()
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.eval()
    optimizer.zero_grad(set_to_none=True)
    losses = model(images, heat.detach(), targets=targets)
    loss = losses['total_loss']
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss')
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError('Nonfinite gradients')
    optimizer.step()
    return {key: float(value.detach()) if torch.is_tensor(value) else float(value)
            for key,value in losses.items()}
