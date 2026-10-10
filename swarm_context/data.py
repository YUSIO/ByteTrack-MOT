"""Sequences of cached detections with labels, causal windows of boxes, batching (Exp066).

A sequence is a detection cache (MOT det.txt rows: frame,-1,x,y,w,h,score) with its annotations. Labels follow the
usual detection matching: boxes are taken by decreasing score and each takes the free annotation with the highest IoU.
  1   matched with IoU >= 0.5
  0   no annotation with IoU >= 0.3, or a duplicate of an already matched annotation (IoU >= 0.5)
 -1   ignored: best IoU in [0.3, 0.5)
"""
import configparser
from pathlib import Path

import numpy as np
import torch


def iou_matrix(a, b):
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, :2] + a[:, None, 2:], b[None, :, :2] + b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    return inter / (np.prod(a[:, 2:], axis=1)[:, None] + np.prod(b[:, 2:], axis=1)[None, :] - inter + 1e-9)


def label_frame(det, gt):
    """det: [n, 5] x,y,w,h,score; gt: [m, 4]. Returns labels [n] and the matched annotation index (or -1)."""
    lab, who = np.zeros(len(det), np.int64), -np.ones(len(det), np.int64)
    if len(det) == 0 or len(gt) == 0:
        return lab, who
    m = iou_matrix(det[:, :4], gt)
    free = np.ones(len(gt), bool)
    for i in np.argsort(-det[:, 4]):
        v = np.where(free, m[i], -1)
        j = int(v.argmax())
        if v[j] >= 0.5:
            lab[i], who[i], free[j] = 1, j, False
        elif m[i].max() >= 0.5:
            lab[i] = 0
        elif m[i].max() >= 0.3:
            lab[i] = -1
    return lab, who


class Sequence:
    def __init__(self, name, det_file, seq_dir, min_score=0.01, max_per_frame=64, frames=None, labelled=True):
        ini = configparser.ConfigParser()
        ini.read(Path(seq_dir) / "seqinfo.ini")
        s = ini["Sequence"]
        self.name, self.width, self.height, self.length = name, int(s["imWidth"]), int(s["imHeight"]), int(s["seqLength"])
        raw = np.loadtxt(det_file, delimiter=",", ndmin=2)
        gt = np.loadtxt(Path(seq_dir) / "gt" / "gt.txt", delimiter=",", ndmin=2) if labelled else np.zeros((0, 6))
        lo, hi = frames if frames else (1, self.length)
        self.first, self.last = lo, hi
        self.det, self.lab, self.gid, self.row = {}, {}, {}, {}
        for f in range(lo, hi + 1):
            idx = np.flatnonzero((raw[:, 0] == f) & (raw[:, 6] >= min_score))
            idx = idx[np.argsort(-raw[idx, 6])][:max_per_frame]
            d = raw[idx][:, 2:7].astype(np.float32)
            g = gt[gt[:, 0] == f]
            lab, who = label_frame(d, g[:, 2:6]) if labelled else (np.full(len(d), -1), -np.ones(len(d), np.int64))
            self.det[f], self.lab[f], self.row[f] = d, lab, idx
            self.gid[f] = np.where(who >= 0, g[np.clip(who, 0, None), 1].astype(np.int64) if len(g) else -1, -1)
        self.raw = raw

    def window(self, t, k, stride=1):
        """Boxes of frames t - k*stride .. t (those that exist). Returns arrays over all tokens of the window."""
        fs = [f for f in range(t - k * stride, t + 1, stride) if self.first <= f <= self.last]
        box = np.concatenate([self.det[f] for f in fs]) if fs else np.zeros((0, 5), np.float32)
        step = np.concatenate([np.full(len(self.det[f]), (f - t) // stride) for f in fs]) if fs else np.zeros(0)
        lab = np.concatenate([self.lab[f] for f in fs]) if fs else np.zeros(0)
        return {"box": box, "step": step.astype(np.int64), "lab": lab.astype(np.int64), "wh": np.array([self.width, self.height], np.float32)}


def augment(w, rng, flip=0.5, drop=0.05, score_noise=0.15):
    box = w["box"].copy()
    if rng.random() < flip:
        box[:, 0] = w["wh"][0] - box[:, 0] - box[:, 2]
    if len(box):
        logit = np.log(np.clip(box[:, 4], 1e-4, 1 - 1e-4) / (1 - np.clip(box[:, 4], 1e-4, 1 - 1e-4))) + rng.normal(0, score_noise, len(box))
        box[:, 4] = 1 / (1 + np.exp(-logit))
        keep = rng.random(len(box)) >= drop
        return {"box": box[keep], "step": w["step"][keep], "lab": w["lab"][keep], "wh": w["wh"]}
    return dict(w, box=box)


def collate(windows):
    n = max(1, max(len(w["box"]) for w in windows))
    b = len(windows)
    box, step = torch.zeros(b, n, 5), torch.zeros(b, n, dtype=torch.long)
    lab, valid, wh = torch.full((b, n), -1, dtype=torch.long), torch.zeros(b, n, dtype=torch.bool), torch.zeros(b, 2)
    for i, w in enumerate(windows):
        m = len(w["box"])
        box[i, :m], step[i, :m], lab[i, :m], valid[i, :m] = torch.from_numpy(w["box"]), torch.from_numpy(w["step"]), torch.from_numpy(w["lab"]), True
        wh[i] = torch.from_numpy(w["wh"])
    return {"box": box, "step": step, "lab": lab, "valid": valid, "wh": wh}


class Windows(torch.utils.data.Dataset):
    def __init__(self, sequences, k, train, seed=0, stride2=0.3):
        self.seqs, self.k, self.train, self.stride2 = sequences, k, train, stride2
        self.index = [(i, f) for i, s in enumerate(sequences) for f in range(s.first, s.last + 1)]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, n):
        i, f = self.index[n]
        if not self.train:
            return self.seqs[i].window(f, self.k)
        stride = 2 if self.rng.random() < self.stride2 else 1
        return augment(self.seqs[i].window(f, self.k, stride), self.rng)
