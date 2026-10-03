"""Causal, sequence-disjoint training windows; test GT is not a model input."""
import configparser
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MEAN = torch.tensor([.485, .456, .406])[:, None, None]
STD = torch.tensor([.229, .224, .225])[:, None, None]


def crop_tensor(image, boxes):
    values = []
    width, height = image.size
    for x, y, w, h in boxes:
        # Exact enclosing integer crop, clamped to the image; no test-driven filter.
        left, top = max(0, min(width - 1, int(np.floor(x)))), max(0, min(height - 1, int(np.floor(y))))
        right, bottom = max(left + 1, min(width, int(np.ceil(x + w)))), max(top + 1, min(height, int(np.ceil(y + h))))
        patch = image.crop((left, top, right, bottom)).resize((128, 128), Image.Resampling.BILINEAR)
        a = torch.from_numpy(np.array(patch, dtype=np.float32).transpose(2, 0, 1)) / 255.
        values.append((a - MEAN) / STD)
    return torch.stack(values) if values else torch.empty((0, 3, 128, 128))


def sequence_info(sequence):
    ini = configparser.ConfigParser()
    ini.read(sequence / 'seqinfo.ini')
    return ini['Sequence']


def image_path(sequence, frame):
    info = sequence_info(sequence)
    return sequence / info.get('imDir', 'img1') / (f'{frame:06d}' + info.get('imExt', '.jpg'))


class Windows(Dataset):
    def __init__(self, root, sequences, window=8, stride=8):
        self.root, self.window = Path(root), window
        self.frames, self.samples = {}, []
        for name in sequences:
            seq = self.root / 'train' / name
            rows = np.loadtxt(seq / 'gt/gt.txt', delimiter=',', ndmin=2)
            rows = rows[(rows[:, 6] >= 1) & (rows[:, 4] > 0) & (rows[:, 5] > 0)]
            frame_map = {int(f): r for f in np.unique(rows[:, 0]) if len(r := rows[rows[:, 0] == f])}
            for f, r in frame_map.items():
                if len(set(r[:, 1])) != len(r):
                    raise ValueError(f'duplicate GT identity {name}:{f}')
            self.frames[name] = frame_map
            length = int(sequence_info(seq)['seqLength'])
            for f in range(window, length + 1, stride):
                if f not in frame_map:
                    continue
                hist_ids = set(np.concatenate([frame_map[h][:, 1] for h in range(f-window+1, f) if h in frame_map])) if any(h in frame_map for h in range(f-window+1, f)) else set()
                if hist_ids.intersection(frame_map[f][:, 1]):
                    self.samples.append((name, f))
        if not self.samples:
            raise ValueError('no eligible windows')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        name, f = self.samples[index]
        fm = self.frames[name]
        order = [f] + [h for h in range(f-self.window+1, f) if h in fm]
        crops, ids, frames = [], [], []
        for h in order:
            r = fm[h]
            with Image.open(image_path(self.root / 'train' / name, h)) as im:
                crops.append(crop_tensor(im.convert('RGB'), r[:, 2:6]))
            ids.extend(r[:, 1].astype(int))
            frames.extend([h] * len(r))
        nc = len(fm[f])
        return {'crops':torch.cat(crops), 'n_current':nc, 'current_ids':torch.tensor(ids[:nc]), 'history_ids':torch.tensor(ids[nc:]), 'history_frames':torch.tensor(frames[nc:]), 'sequence':name, 'frame':f}
