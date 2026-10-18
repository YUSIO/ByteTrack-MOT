"""Adjacent-frame pairs of MOT-format sequences for training the temporal fusion (Exp091).

Each item: letterboxed frames t-1 and t, the camera-motion affine t-1 -> t in letterbox-input coordinates (identity
when the file has no row), and the frame-t labels as YOLOX expects them ([class, cx, cy, w, h] in input pixels, padded
to `max_labels` rows). A pair is flipped horizontally as a whole (both images, labels and the affine).
"""
import configparser
from pathlib import Path

import cv2
import numpy as np
import torch

from yolox.data.data_augment import preproc

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def seq_info(seq_dir):
    ini = configparser.ConfigParser()
    ini.read(seq_dir / "seqinfo.ini")
    s = ini["Sequence"]
    return int(s["seqLength"]), int(s["imWidth"]), int(s["imHeight"]), s.get("imDir", "img1"), s.get("imExt", ".jpg")


def load_gmc(path):
    warps = {}
    if Path(path).exists():
        for r in np.loadtxt(path, ndmin=2):
            warps[int(r[0])] = np.vstack([r[1:].reshape(2, 3), [0, 0, 1]])
    return warps


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, view, sequences, gmc_dir, size=(1088, 1088), max_labels=120, flip=0.5, seed=0):
        self.view, self.size, self.max_labels, self.flip = Path(view), size, max_labels, flip
        self.items, self.meta, self.gt, self.gmc = [], {}, {}, {}
        for s in sequences:
            length, w, h, img_dir, ext = seq_info(self.view / s)
            self.meta[s] = (w, h, img_dir, ext)
            gt = np.loadtxt(self.view / s / "gt" / "gt.txt", delimiter=",", ndmin=2)
            gt = gt[gt[:, 6] >= 1] if gt.size else np.zeros((0, 9))
            self.gt[s] = {int(f): gt[gt[:, 0] == f][:, 2:6] for f in np.unique(gt[:, 0])}
            self.gmc[s] = load_gmc(Path(gmc_dir) / f"{s}.txt")
            self.items += [(s, t) for t in range(2, length + 1)]
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.items)

    def image(self, s, f):
        w, h, img_dir, ext = self.meta[s]
        img = cv2.imread(str(self.view / s / img_dir / f"{f:06d}{ext}"))
        assert img is not None, (s, f)
        return img

    def __getitem__(self, i):
        s, t = self.items[i]
        w, h = self.meta[s][:2]
        prev, cur = self.image(s, t - 1), self.image(s, t)
        boxes = self.gt[s].get(t, np.zeros((0, 4))).copy()
        m = self.gmc[s].get(t, np.eye(3)).copy()
        rng = np.random.default_rng([self.seed, self.epoch, i])  # per item and epoch: independent of loader workers
        if rng.random() < self.flip:
            prev, cur = prev[:, ::-1], cur[:, ::-1]
            boxes[:, 0] = w - boxes[:, 0] - boxes[:, 2]
            fl = np.array([[-1.0, 0, w], [0, 1, 0], [0, 0, 1]])
            m = fl @ m @ fl
        x0, r = preproc(np.ascontiguousarray(prev), self.size, MEAN, STD)
        x1, _ = preproc(np.ascontiguousarray(cur), self.size, MEAN, STD)
        sc = np.diag([r, r, 1.0])
        m_in = sc @ m @ np.linalg.inv(sc)
        x1b = np.clip(boxes[:, 0], 0, w); y1b = np.clip(boxes[:, 1], 0, h)
        x2b = np.clip(boxes[:, 0] + boxes[:, 2], 0, w); y2b = np.clip(boxes[:, 1] + boxes[:, 3], 0, h)
        keep = (x2b - x1b > 1) & (y2b - y1b > 1)
        lab = np.zeros((self.max_labels, 5), np.float32)
        b = np.stack([(x1b + x2b) / 2, (y1b + y2b) / 2, x2b - x1b, y2b - y1b], 1)[keep] * r
        n = min(len(b), self.max_labels)
        lab[:n, 1:] = b[:n]
        return (torch.from_numpy(x0), torch.from_numpy(x1), torch.from_numpy(np.linalg.inv(m_in).astype(np.float32)), torch.from_numpy(lab))
