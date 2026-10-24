"""Explicit train-only teacher-forcing data; never imported by online inference."""
import configparser
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from yolox.data.data_augment import preproc

MEAN, STD = (.485, .456, .406), (.229, .224, .225)


def sequence_info(path):
    cfg = configparser.ConfigParser()
    cfg.read(Path(path) / 'seqinfo.ini')
    return dict(cfg['Sequence'])


def read_image(path, frame, info, size):
    filename = path / info.get('imdir', 'img1') / f"{frame:06d}{info.get('imext', '.jpg')}"
    image = cv2.imread(str(filename))
    if image is None:
        raise FileNotFoundError(filename)
    x, ratio = preproc(image, tuple(size), MEAN, STD)
    return torch.from_numpy(x), ratio, image.shape[:2]


def read_train_sequences(root, sequences):
    root = Path(root)
    if not sequences or len(sequences) != len(set(sequences)):
        raise ValueError('An explicit nonempty unique training sequence list is required')
    data = {}
    for name in sequences:
        if Path(name).name != name:
            raise ValueError('Sequence names may not contain paths')
        path = root / 'train' / name
        info = sequence_info(path)
        rows = np.loadtxt(path / 'gt' / 'gt.txt', delimiter=',', ndmin=2)
        frames = {i: {} for i in range(1, int(info['seqlength']) + 1)}
        for row in rows:
            if row[6] < 1:
                continue
            frame, identity = int(row[0]), int(row[1])
            if identity in frames[frame]:
                raise ValueError(f'Duplicate GT ID: {name}/{frame}/{identity}')
            frames[frame][identity] = row[2:6].astype(np.float32)
        data[name] = (path, info, frames)
    return data


def reliable_centers(frames, end, window):
    if end < window:
        return np.empty((0, window, 2), np.float32)
    selected = [frames[i] for i in range(end - window + 1, end + 1)]
    ids = sorted(set.intersection(*(set(f) for f in selected)))
    return np.asarray([[f[i][:2] + f[i][2:] / 2 for f in selected] for i in ids],
                      np.float32).reshape(-1, window, 2)


class DetectionPairs(Dataset):
    def __init__(self, root, sequences, size, window=8):
        self.data = read_train_sequences(root, sequences)
        self.size, self.window = size, window
        self.index = [(name, t) for name, (_, info, _) in self.data.items()
                      for t in range(1, int(info['seqlength']) + 1)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        name, frame = self.index[index]
        path, info, frames = self.data[name]
        image, ratio, shape = read_image(path, frame, info, self.size)
        previous, old_ratio, old_shape = read_image(path, max(1, frame-1), info, self.size)
        if old_shape != shape or ratio != old_ratio:
            raise ValueError('Pair geometry changed')
        boxes = np.asarray(list(frames[frame].values()), np.float32).reshape(-1, 4)
        targets = np.zeros((len(boxes), 5), np.float32)
        targets[:, 1:3] = (boxes[:, :2] + boxes[:, 2:] / 2) * ratio
        targets[:, 3:] = boxes[:, 2:] * ratio
        last = frames[frame-1] if frame > 1 else {}
        ids = sorted(last)
        return {'image': image, 'previous': previous, 'targets': torch.from_numpy(targets),
                'boxes': torch.tensor(np.asarray([last[i] for i in ids], np.float32).reshape(-1,4)) * ratio,
                'ids': torch.tensor(ids, dtype=torch.long),
                'history': torch.from_numpy(reliable_centers(frames, frame-1, self.window)) * ratio,
                'sequence': name, 'frame': frame, 'prior_source': 'train_gt_history'}


class TrajectoryWindows(Dataset):
    def __init__(self, root, sequences, size, observed=8, predicted=12):
        self.data = read_train_sequences(root, sequences)
        self.size, self.observed, self.predicted = size, observed, predicted
        self.index = []
        total = observed + predicted
        for name, (_, info, frames) in self.data.items():
            for end in range(total, int(info['seqlength']) + 1):
                centers = reliable_centers(frames, end, total)
                if len(centers):
                    self.index.append((name, end))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        name, end = self.index[index]
        _, info, frames = self.data[name]
        scale = min(self.size[0] / int(info['imheight']), self.size[1] / int(info['imwidth']))
        centers = torch.from_numpy(reliable_centers(frames, end, self.observed+self.predicted)) * scale
        t = self.observed
        return {'history': centers[:, :t], 'future': centers[:, t:],
                'reference': centers[:, t-3:t+4].mean(1), 'sequence': name, 'end': end}
