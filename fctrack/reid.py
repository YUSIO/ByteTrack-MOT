"""Fixed crop ReID probe for full A; GT appears only in fit-only training.

Association borrows BoT-SORT's proximity-gated min(IoU-score, cosine/2).
Only detector observations update the EMA; a recovery proposal has no feature.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def crop(image, box, size=64):
    h, w = image.shape[:2]
    x, y, bw, bh = box
    x1, y1 = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
    x2, y2 = min(w, int(np.ceil(x + bw))), min(h, int(np.ceil(y + bh)))
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.resize(image[y1:y2, x1:x2, ::-1], (size, size), interpolation=cv2.INTER_LINEAR).copy()


def tensor(crops, device):
    x = torch.as_tensor(np.stack(crops).transpose(0, 3, 1, 2), device=device).float() / 255
    mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
    std = x.new_tensor([.229, .224, .225])[None, :, None, None]
    return (x - mean) / std


class Encoder(nn.Module):
    def __init__(self, classes, initial=None):
        super().__init__()
        self.backbone = resnet18(weights=None)
        if initial is not None:
            self.backbone.load_state_dict(torch.load(initial, map_location='cpu', weights_only=True))
        self.backbone.fc = nn.Identity()
        self.project = nn.Linear(512, 128, bias=False)
        self.neck = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, classes, bias=False)

    def forward(self, x):
        z = self.neck(self.project(self.backbone(x)))
        return F.normalize(z, dim=1), self.classifier(z)


def train(root, cfg, output, device):
    """Fixed last-epoch selection; no dev GT, dev crops, or dev metrics read."""
    from torch.utils.tensorboard import SummaryWriter
    spec = cfg['reid_training']
    rng = np.random.default_rng(cfg['seed'])
    crops, labels, identities = [], [], {}
    sequence_labels = []
    for name in cfg['fit_sequences']:
        seq = root / cfg['cell']['dataset_root'] / 'train' / name
        gt = np.loadtxt(seq / 'gt/gt.txt', delimiter=',', ndmin=2)
        gt = gt[(gt[:, 6] >= 1) & (gt[:, 4] > 0) & (gt[:, 5] > 0)]
        ids = []
        for frame in sorted(set(gt[:, 0].astype(int))):
            im = cv2.imread(str(seq / 'img1' / f'{frame:06d}.jpg'))
            assert im is not None
            for row in gt[gt[:, 0] == frame]:
                c = crop(im, row[2:6])
                if c is None:
                    continue
                key = (name, int(row[1]))
                if key not in identities:
                    identities[key] = len(identities)
                    ids.append(identities[key])
                crops.append(c)
                labels.append(identities[key])
        sequence_labels.append(ids)
        print('fit_crops', name, len(crops), flush=True)
    crops, labels = np.stack(crops), np.asarray(labels)
    by_id = {i: np.flatnonzero(labels == i) for i in range(len(identities))}
    model = Encoder(len(identities), root / spec['initial_weights']).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=spec['lr'], weight_decay=spec['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, spec['epochs'] * spec['steps_per_epoch'])
    writer = SummaryWriter(str(output / 'tensorboard'))
    history = []
    for epoch in range(spec['epochs']):
        sums = np.zeros(4)
        for step in range(spec['steps_per_epoch']):
            chosen = rng.choice(len(sequence_labels), 4, replace=False)
            ids = np.concatenate([rng.choice(sequence_labels[s], min(4, len(sequence_labels[s])), replace=False) for s in chosen])
            ii = np.concatenate([rng.choice(by_id[int(i)], 4, replace=len(by_id[int(i)]) < 4) for i in ids])
            batch = crops[ii].copy()
            flip = rng.random(len(batch)) < .5
            batch[flip] = batch[flip, :, ::-1]
            x, y = tensor(batch, device), torch.as_tensor(labels[ii], device=device)
            z, logits = model(x)
            ce = F.cross_entropy(logits, y, label_smoothing=.1)
            dist = torch.cdist(z, z)
            same = y[:, None] == y[None, :]
            same.fill_diagonal_(False)
            positive = dist.masked_fill(~same, -1).max(1).values
            negative = dist.masked_fill(y[:, None] == y[None, :], 10).min(1).values
            triplet = F.relu(positive - negative + spec['margin']).mean()
            loss = ce + triplet
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            scheduler.step()
            sums += [loss.item(), ce.item(), triplet.item(), (logits.argmax(1) == y).float().mean().item()]
        row = dict(epoch=epoch + 1, **dict(zip(['loss', 'ce', 'triplet', 'accuracy'], (sums / spec['steps_per_epoch']).tolist())))
        history.append(row)
        for k, v in row.items():
            if k != 'epoch':
                writer.add_scalar('train/' + k, v, epoch + 1)
        writer.flush()
        print('epoch', json.dumps(row), flush=True)
    writer.close()
    torch.save({'state_dict': model.cpu().state_dict(), 'classes': len(identities), 'seed': cfg['seed'], 'config': spec}, output / 'last.pt')
    metrics = dict(history=history, classes=len(identities), crops=len(crops), parameters=sum(p.numel() for p in model.parameters()), selection='fixed_last_epoch', fit_sequences=cfg['fit_sequences'])
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')


@torch.inference_mode()
def export(root, cfg, output, checkpoint, device):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = Encoder(ck['classes']).to(device).eval()
    model.load_state_dict(ck['state_dict'])
    stats = {}
    for name in cfg['dev_sequences']:
        detpath = root / cfg['cell']['detections'] / name / 'det.txt'
        raw = np.loadtxt(detpath, delimiter=',', ndmin=2)
        # Match BYTETracker's float32 confidence comparison exactly.
        indices = np.flatnonzero(raw[:, 6].astype(np.float32) > cfg['tracker']['track_thresh'])
        selected = raw[indices]
        vectors = np.zeros((len(indices), 128), np.float32)
        valid = np.zeros(len(indices), bool)
        for frame in np.unique(selected[:, 0]).astype(int):
            im = cv2.imread(str(root / cfg['cell']['dataset_root'] / 'train' / name / 'img1' / f'{frame:06d}.jpg'))
            assert im is not None
            jj, cc = [], []
            for j in np.flatnonzero(selected[:, 0] == frame):
                c = crop(im, selected[j, 2:6].astype(np.float32))
                if c is not None:
                    jj.append(j)
                    cc.append(c)
            if cc:
                vectors[jj] = model(tensor(cc, device))[0].cpu().numpy()
                valid[jj] = True
        np.savez_compressed(output / f'{name}.npz', row_indices=indices, vectors=vectors, valid=valid, detection_sha256=sha(detpath), checkpoint_sha256=sha(checkpoint))
        stats[name] = {'high_detections': len(indices), 'valid_crops': int(valid.sum())}
        print('features', name, stats[name], flush=True)
    (output / 'metrics.json').write_text(json.dumps(stats, indent=2) + '\n')


def fuse(tracks, detections, geometry_cost, spec):
    """Only replace nearby, appearance-qualified entries; missing features are neutral."""
    from yolox.tracker import matching
    cost = geometry_cost.copy()
    if not tracks or not detections:
        return cost
    iou = matching.iou_distance(tracks, detections)
    for i, track in enumerate(tracks):
        a = getattr(track, 'reid_feature', None)
        if a is None:
            continue
        for j, det in enumerate(detections):
            b = getattr(det, 'reid_feature', None)
            if b is None or iou[i, j] > spec['proximity']:
                continue
            d = max(0., 1. - float(np.dot(a, b))) / 2
            if d <= spec['appearance']:
                cost[i, j] = min(cost[i, j], d)
    return cost
