"""Training samples of the track head from annotated trajectories (training sequences only).

One item = one current frame t of one sequence. A gap k (1 with probability 0.5, otherwise uniform in 2..5) gives the
reference frame r = t - k, the frame the tracks were last observed in; the template frame u = r - m (m = 0 with
probability 0.5, otherwise uniform in 1..10) is where the track's appearance was last taken from a detector box.
Every member annotated at r becomes a query. Its state is built from the annotations with noise, the way the tracker
builds it at inference:

  last box      the annotated box at r, centre jittered by 0.05 side, size by exp(N(0, 0.05))
  motion        mean displacement over up to 3 annotated steps before r (jittered); prediction = centre + k * velocity
  formation     the other members annotated at r and t, each kept with probability 0.8 ("tracked now"), their centres
                at t jittered; up to 6 Delaunay neighbours give the offset estimates (fctrack.formation)
  placement     the formation estimate, or the motion prediction when no neighbour is left (the Exp070 fallback rule),
                plus N(0, s * side) with s ~ U(0, 0.5); with probability 0.15 it is put on another member that is
                tracked now (within 4 sides) instead, which makes "is it this track's UAV or a neighbour" a question
  label         the member's annotated box at t if its centre is inside the window, otherwise absent

Only the frames listed in `frames` are used as t, r and u, so that the detector features come from frames the
detector did not train on (the second temporal half of every MOT-train sequence, Exp043 `val.json`).
"""
import configparser
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from fctrack.formation import estimate, select_neighbours
from jdt.detector import letterbox
from jdt.query import make_query


def load_gt(seq_dir):
    g = np.loadtxt(Path(seq_dir) / "gt/gt.txt", delimiter=",", ndmin=2)
    by_frame = defaultdict(dict)
    for row in g:
        by_frame[int(row[0])][int(row[1])] = row[2:6].astype(np.float64)
    return by_frame


def seq_size(seq_dir):
    ini = configparser.ConfigParser()
    ini.read(Path(seq_dir) / "seqinfo.ini")
    return int(ini["Sequence"]["imWidth"]), int(ini["Sequence"]["imHeight"])


def centre(b):
    return b[:2] + b[2:] / 2


class TrackSamples(Dataset):
    def __init__(self, seq_dirs, frames, train=True, seed=0, max_queries=24):
        """seq_dirs: sequence directories; frames: {sequence name: sorted list of usable frames}."""
        self.train, self.seed, self.max_queries = train, seed, max_queries
        self.seqs, self.items = [], []
        for d in seq_dirs:
            d = Path(d)
            fr = sorted(frames[d.name])
            gt = load_gt(d)
            self.seqs.append({"dir": d, "gt": gt, "size": seq_size(d), "first": fr[0], "frames": set(fr)})
            s = len(self.seqs) - 1
            self.items += [(s, t) for t in fr if t - 1 in self.seqs[s]["frames"]]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s, t = self.items[i]
        S = self.seqs[s]
        rng = np.random.default_rng(None if self.train else self.seed * 1000003 + i)
        gt, (W, H), first = S["gt"], S["size"], S["first"]
        k = 1 if rng.random() < 0.5 else int(rng.integers(2, 6))
        r = max(t - k, first)
        k = t - r
        m = 0 if rng.random() < 0.5 else int(rng.integers(1, 11))
        u = max(r - m, first)
        flip = bool(self.train and rng.random() < 0.5)
        at_r, at_t = gt.get(r, {}), gt.get(t, {})
        ids_r = list(at_r)
        P0 = np.array([centre(at_r[j]) for j in ids_r]).reshape(-1, 2)
        sides_r = np.array([np.sqrt(at_r[j][2] * at_r[j][3]) for j in ids_r])
        kept = {j: (j in at_t) and rng.random() < 0.8 for j in ids_r}
        now = {j: centre(at_t[j]) + rng.normal(0, 0.05, 2) * np.sqrt(at_t[j][2] * at_t[j][3]) for j in at_t if kept.get(j, rng.random() < 0.8)}
        ratio = min(1088 / H, 1088 / W)  # jdt.detector.letterbox
        order = rng.permutation(len(ids_r))[: self.max_queries]
        queries = []
        for a in order:
            j = ids_r[a]
            b = at_r[j].copy()
            side = sides_r[a]
            b[:2] += rng.normal(0, 0.05, 2) * side
            b[2:] *= np.exp(rng.normal(0, 0.05, 2))
            c_r = centre(b)
            hist = [centre(gt[f][j]) for f in range(r - 3, r + 1) if j in gt.get(f, {})]
            vel = np.diff(np.array(hist), axis=0).mean(0) if len(hist) >= 2 else np.zeros(2)
            vel = vel + rng.normal(0, 0.05, 2) * side
            cv = c_r + k * vel
            avail = np.array([ids_r[x] in now and ids_r[x] != j for x in range(len(ids_r))])
            nb = select_neighbours(P0, int(a), avail, "delaunay", 6) if avail.any() else []
            if nb:
                N0 = P0[nb] - P0[a] + c_r
                N1 = np.array([now[ids_r[x]] for x in nb])
                est, spread, _ = estimate(c_r, N0, N1, "offset", 0.5 * side)
                each = N1 + (c_r - N0)
                place = est.copy()
            else:
                each, spread, place = np.zeros((0, 2)), 0.0, cv.copy()
            taken = np.array([p for o, p in now.items() if o != j]).reshape(-1, 2)
            if self.train and rng.random() < 0.15 and len(taken):
                close = taken[np.linalg.norm(taken - place, axis=1) < 4 * side]
                if len(close):
                    place = close[int(rng.integers(len(close)))].copy()
            else:
                place = place + rng.normal(0, rng.uniform(0, 0.5), 2) * side
            tb = gt.get(u, {}).get(j)
            tb = (tb.copy() if tb is not None else at_r[j].copy())
            tb[:2] += rng.normal(0, 0.05, 2) * np.sqrt(tb[2] * tb[3])
            g = at_t.get(j)
            g = None if g is None else np.r_[centre(g), g[2:]]
            queries.append(make_query(centre=place, box=b, tbox=tb, cv=cv, vel=vel, gap=k, nb=each, spread=spread, occ=taken,
                                      member_sides=sides_r, W=W, H=H, r=ratio, flip=flip, gt=g))
        img_t, img_u = self.image(S, t, flip), self.image(S, u, flip)
        lt, rt = letterbox(img_t)
        lu, _ = letterbox(img_u)
        assert abs(rt - ratio) < 1e-12
        return lt, lu, queries

    def image(self, S, f, flip):
        img = cv2.imread(str(S["dir"] / "img1" / f"{f:06d}.jpg"))
        return img[:, ::-1].copy() if flip else img


def collate(batch):
    import torch
    B = len(batch)
    imgs = torch.from_numpy(np.stack([b[0] for b in batch] + [b[1] for b in batch]))
    qs, qimg = [], []
    for i, b in enumerate(batch):
        qs += b[2]
        qimg += [i] * len(b[2])
    return imgs, qs, torch.tensor(qimg), B
