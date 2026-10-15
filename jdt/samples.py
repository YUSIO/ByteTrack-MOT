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
  state error   the track's own centre at r is jittered by N(0, s side), s ~ U(0, 0.5); with probability 0.15 it sits
                on another member within 4 sides (drifted to a neighbour), with probability 0.2 it is displaced by 2-5
                sides; extra queries are tracks born on clutter (see TrackSamples.state / background)
  placement     the formation estimate from that state, or the motion prediction when no neighbour is left (the
                Exp070 fallback rule), exactly as at inference: the window never moves relative to the state, so the
                structure cues cannot reveal the label through the placement; windows whose box overlaps a member
                tracked now (IoU >= 0.3) are dropped, as the tracker never looks there
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
from fctrack.tracker import iou_matrix
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
    def __init__(self, seq_dirs, frames, train=True, seed=0, max_queries=24, labels=False):
        """seq_dirs: sequence directories; frames: {sequence name: sorted list of usable frames}; labels: also return
        the annotated boxes of the current frame (network-input cx, cy, w, h) for a detection loss."""
        self.train, self.seed, self.max_queries, self.labels = train, seed, max_queries, labels
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
            side = sides_r[a]
            true_c = P0[a]
            state_c, kind = self.state(rng, true_c, P0, a, side)  # where the tracker believes the track was at r
            shift = state_c - true_c
            b = at_r[j].copy()
            b[:2] += shift
            b[2:] *= np.exp(rng.normal(0, 0.05, 2))
            hist = [centre(gt[f][j]) for f in range(r - 3, r + 1) if j in gt.get(f, {})]
            vel = np.diff(np.array(hist), axis=0).mean(0) if len(hist) >= 2 else np.zeros(2)
            vel = vel + rng.normal(0, 0.05, 2) * side
            cv = state_c + k * vel
            Ps = P0.copy()
            Ps[a] = state_c
            avail = np.array([ids_r[x] in now and ids_r[x] != j for x in range(len(ids_r))])
            nb = select_neighbours(Ps, int(a), avail, "delaunay", 6) if avail.any() else []
            if nb:
                N1 = np.array([now[ids_r[x]] for x in nb])
                place, spread, _ = estimate(state_c, Ps[nb], N1, "offset", 0.5 * side)
                each = N1 + (state_c - Ps[nb])
            else:
                each, spread, place = np.zeros((0, 2)), 0.0, cv.copy()
            taken = np.array([p for o, p in now.items() if o != j]).reshape(-1, 2)
            if self.on_taken(place, b[2:], now, at_t, j):
                continue
            tb = gt.get(u, {}).get(j)
            tb = (tb.copy() if tb is not None else at_r[j].copy())
            tb[:2] += rng.normal(0, 0.05, 2) * np.sqrt(tb[2] * tb[3]) + (shift if kind == 2 else 0)
            g = at_t.get(j)
            g = None if g is None else np.r_[centre(g), g[2:]]
            q = make_query(centre=place, box=b, tbox=tb, cv=cv, vel=vel, gap=k, nb=each, spread=spread, occ=taken,
                           member_sides=sides_r, W=W, H=H, r=ratio, flip=flip, gt=g)
            q["kind"] = kind
            queries.append(q)
        queries += self.background(rng, at_r, at_t, P0, sides_r, now, k, W, H, ratio, flip, max(1, int(round(0.3 * len(order)))) if len(ids_r) else 0)
        img_t, img_u = self.image(S, t, flip), self.image(S, u, flip)
        lt, rt = letterbox(img_t)
        lu, _ = letterbox(img_u)
        assert abs(rt - ratio) < 1e-12
        if self.labels:
            boxes = np.array([np.r_[centre(g), g[2:]] for g in at_t.values()]).reshape(-1, 4)
            if flip:
                boxes[:, 0] = W - boxes[:, 0]
            return lt, lu, queries, boxes * ratio
        return lt, lu, queries

    @staticmethod
    def on_taken(place, wh, now, at_t, own):
        """The tracker never looks where its estimated box overlaps a member tracked in this frame (IoU >= 0.3)."""
        others = [o for o in now if o != own]
        if not others:
            return False
        tb = np.array([np.r_[now[o] - at_t[o][2:] / 2, at_t[o][2:]] for o in others])
        return iou_matrix(np.r_[place - wh / 2, wh], tb).max() >= 0.3

    @staticmethod
    def state(rng, true_c, P0, a, side):
        """The track's own centre at r as the tracker holds it. The window is then placed by the inference rule
        (formation estimate, else constant velocity), so no sample can be told apart by where the window sits
        relative to the formation or motion cues. kind 0: N(0, s side), s ~ U(0, 0.5); 1 (p 0.15): the track sits on
        another member within 4 sides (it drifted to a neighbour); 2 (p 0.2): displaced by 2-5 sides (template too)."""
        u = rng.random()
        if u < 0.15:
            d = np.linalg.norm(P0 - true_c, axis=1)
            close = [x for x in range(len(P0)) if x != a and d[x] < 4 * side]
            if close:
                return P0[close[int(rng.integers(len(close)))]] + rng.normal(0, 0.05, 2) * side, 1
        if u < 0.35:
            ang = rng.uniform(0, 2 * np.pi)
            return true_c + rng.uniform(2, 5) * side * np.array([np.cos(ang), np.sin(ang)]), 2
        return true_c + rng.normal(0, rng.uniform(0, 0.5), 2) * side, 0

    def background(self, rng, at_r, at_t, P0, sides_r, now, k, W, H, ratio, flip, n):
        """kind 3: tracks that never were a UAV (a track born on clutter): a box at a place with no annotated UAV
        within 3 sides at r and t, a small random velocity, formation offsets to the real members; always absent."""
        out = []
        C_t = np.array([centre(b) for b in at_t.values()]).reshape(-1, 2)
        for _ in range(n):
            a = int(rng.integers(len(P0)))
            src = list(at_r.values())[a]
            side = sides_r[a]
            for _ in range(20):
                c = np.array([rng.uniform(0.02, 0.98) * W, rng.uniform(0.02, 0.98) * H])
                far = lambda C: len(C) == 0 or np.linalg.norm(C - c, axis=1).min() > 3 * side
                if far(P0) and far(C_t):
                    break
            else:
                continue
            w, h = src[2] * np.exp(rng.normal(0, 0.1)), src[3] * np.exp(rng.normal(0, 0.1))
            b = np.array([c[0] - w / 2, c[1] - h / 2, w, h])
            vel = rng.normal(0, 0.3, 2) * side
            cv = c + k * vel
            ids = list(at_r)
            Pb = np.vstack((P0, c))
            avail = np.array([i in now for i in ids] + [False])
            nb = select_neighbours(Pb, len(ids), avail, "delaunay", 6) if avail.any() else []
            if nb:
                N1 = np.array([now[ids[x]] for x in nb])
                est, spread, _ = estimate(c, Pb[nb], N1, "offset", 0.5 * side)
                each, place = N1 + (c - Pb[nb]), est
            else:
                each, spread, place = np.zeros((0, 2)), 0.0, cv
            taken = np.array(list(now.values())).reshape(-1, 2)
            if self.on_taken(place, b[2:], now, at_t, None):
                continue
            q = make_query(centre=place, box=b, tbox=b, cv=cv, vel=vel, gap=k, nb=each, spread=spread, occ=taken,
                           member_sides=sides_r, W=W, H=H, r=ratio, flip=flip, gt=None)
            if len(C_t):  # a window that happens to contain a UAV would teach "the track is that UAV": skip
                cn = (np.array([W - C_t[:, 0], C_t[:, 1]]).T if flip else C_t) * ratio
                if (np.abs(cn - q["centre"]).max(1) <= q["half"]).any():
                    continue
            q["kind"] = 3
            out.append(q)
        return out

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
    if len(batch[0]) == 4:  # detection targets: B x max x (class, cx, cy, w, h), zero rows are padding
        n = max(1, max(len(b[3]) for b in batch))
        tg = torch.zeros(B, n, 5)
        for i, b in enumerate(batch):
            tg[i, : len(b[3]), 1:] = torch.from_numpy(b[3]).float()
        return imgs, qs, torch.tensor(qimg), B, tg
    return imgs, qs, torch.tensor(qimg), B
