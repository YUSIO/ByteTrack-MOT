"""Verification networks for an estimated box, and the training samples for them.

paper   ResNet18 on a 224 x 128 native-resolution crop around the estimated position: UAV or background (V-USDT).
        The estimated box is kept unchanged when the crop is accepted.
paper_resized
        The same ResNet18, but the 224 x 128 input is a crop of PR_ZOOM box sides in height (and 224 / 128 times that in
        width) around the estimated position, resized; the other reading of "cropped 224x128 images around the
        estimated position" in the paper. Not in the paper as such.
seed    A small network on a crop that covers ZOOM x the box side at a fixed 64 x 64 size. It answers whether a UAV
        centre lies within R_POS box sides of the estimated centre, and regresses that UAV's box. Not in the paper.

Training samples are drawn from annotated training frames only; no detector or tracker output is read.
"""
import configparser
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

PAPER_W, PAPER_H = 224, 128
IN, ZOOM, R_POS, R_NEG = 64, 3.0, 1.0, 1.5
PR_ZOOM = 3.0
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def paper_crop(img, cx, cy):
    """224 x 128 native pixels centred at (cx, cy), RGB, normalised, CHW."""
    M = np.array([[1, 0, PAPER_W / 2 - cx], [0, 1, PAPER_H / 2 - cy]], np.float32)
    c = cv2.warpAffine(img, M, (PAPER_W, PAPER_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return ((c[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)


def paper_resized_crop(img, cx, cy, side):
    """A crop of PR_ZOOM x side (height) by PR_ZOOM x side x 224 / 128 (width) around (cx, cy), resized to 224 x 128."""
    s = PAPER_H / (PR_ZOOM * side)
    M = np.array([[s, 0, PAPER_W / 2 - s * cx], [0, s, PAPER_H / 2 - s * cy]], np.float32)
    c = cv2.warpAffine(img, M, (PAPER_W, PAPER_H), flags=cv2.INTER_LINEAR if s >= 1 else cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    return ((c[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)


def seed_crop(img, cx, cy, side, flip=False):
    """IN x IN crop covering ZOOM x side native pixels around (cx, cy), RGB in [0, 1], CHW."""
    s = IN / (ZOOM * side)
    M = np.array([[s, 0, IN / 2 - s * cx], [0, s, IN / 2 - s * cy]], np.float32)
    c = cv2.warpAffine(img, M, (IN, IN), flags=cv2.INTER_LINEAR if s >= 1 else cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    c = c[:, ::-1] if flip else c
    return np.ascontiguousarray(c[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)) / 255.0


class PaperVerifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        import torchvision
        self.net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.net.fc = nn.Linear(self.net.fc.in_features, 1)

    def forward(self, x):
        return self.net(x)[:, 0]


def _block(i, o, s):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SeedVerifier(nn.Module):
    """Outputs (presence logit, dx / side, dy / side, log(w / seed w), log(h / seed h)[, birth logit]).

    Exp112: with birth=True a sixth output answers, for a detection box used as the seed, whether that box is a correct
    birth (IoU >= 0.5 with an annotated UAV); the first five outputs keep their fill-in meaning."""

    def __init__(self, birth=False):
        super().__init__()
        c = (32, 64, 128, 192, 256)
        self.body = nn.Sequential(_block(3, c[0], 1), _block(c[0], c[1], 2), _block(c[1], c[2], 2), _block(c[2], c[3], 2), _block(c[3], c[4], 2))
        self.head = nn.Sequential(nn.Linear(c[4] * 2 + 2, 256), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(256, 6 if birth else 5))

    def forward(self, x, scal):
        f = self.body(x)
        return self.head(torch.cat((f.mean((2, 3)), f.amax((2, 3)), scal), 1))


def scalars(side, width):
    """Absolute scale cues that the scale-normalised crop removes."""
    return np.stack((np.log(side / 20.0), np.log(width / 1000.0)), -1).astype(np.float32)


def load(kind, path, device):
    state = torch.load(path, map_location=device)
    birth = any(k.endswith("head.3.weight") and v.shape[0] == 6 for k, v in state.items())
    model = (PaperVerifier(pretrained=False) if kind.startswith("paper") else SeedVerifier(birth=birth)).to(device)
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def verify(kind, model, img, tlwh, device, head=0):
    """tlwh: n x 4 estimated boxes of one image. Returns (p, boxes): presence and the box to keep for each.
    head = 5 reads the Exp112 birth output instead of the fill-in presence output (seed kind only)."""
    tlwh = np.asarray(tlwh, float).reshape(-1, 4)
    cx, cy = tlwh[:, 0] + tlwh[:, 2] / 2, tlwh[:, 1] + tlwh[:, 3] / 2
    if kind == "paper":
        x = torch.from_numpy(np.stack([paper_crop(img, a, b) for a, b in zip(cx, cy)])).to(device)
        return torch.sigmoid(model(x)).cpu().numpy().astype(float), tlwh.copy()
    side = np.sqrt(tlwh[:, 2] * tlwh[:, 3])
    if kind == "paper_resized":
        x = torch.from_numpy(np.stack([paper_resized_crop(img, a, b, s) for a, b, s in zip(cx, cy, side)])).to(device)
        return torch.sigmoid(model(x)).cpu().numpy().astype(float), tlwh.copy()
    x = torch.from_numpy(np.stack([seed_crop(img, a, b, s) for a, b, s in zip(cx, cy, side)])).to(device)
    out = model(x, torch.from_numpy(scalars(side, np.full(len(side), img.shape[1], float))).to(device)).cpu().numpy().astype(float)
    w, h = tlwh[:, 2] * np.exp(out[:, 3]), tlwh[:, 3] * np.exp(out[:, 4])
    ncx, ncy = cx + out[:, 1] * side, cy + out[:, 2] * side
    return 1 / (1 + np.exp(-out[:, head])), np.column_stack((ncx - w / 2, ncy - h / 2, w, h))


class FrameSamples(Dataset):
    """One item = one annotated frame; returns `per_frame` training samples drawn from it."""

    def __init__(self, sequences, kind, per_frame, train, step=1, seed=0, preload=False, birth=None, birth_per_frame=0):
        """birth (Exp112): {(sequence name, frame): n x 6 array of detection tlwh, label (1 / 0), index of the matched GT}"""
        self.kind, self.per_frame, self.train, self.seed = kind, per_frame, train, seed
        self.birth, self.birth_per_frame = birth or {}, birth_per_frame
        self.frames, self.images = [], None
        for d in sequences:
            d = Path(d)
            g = np.loadtxt(d / "gt/gt.txt", delimiter=",", ndmin=2)
            ini = configparser.ConfigParser()
            ini.read(d / "seqinfo.ini")
            for f in np.unique(g[:, 0])[::step]:
                self.frames.append((str(d / "img1" / f"{int(f):06d}.jpg"), g[g[:, 0] == f][:, 2:6].astype(np.float32), int(ini["Sequence"]["imWidth"]),
                                    self.birth.get((d.name, int(f)))))

        if preload:  # decode every frame once, before the loader forks its workers (they share the memory)
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(8) as pool:
                self.images = list(pool.map(lambda fr: cv2.imread(fr[0]), self.frames))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        path, gt, width, cands = self.frames[i]
        rng = np.random.default_rng(None if self.train else self.seed * 100003 + i)
        img = self.images[i] if self.images is not None else cv2.imread(path)
        H, W = img.shape[:2]
        gc, gs = gt[:, :2] + gt[:, 2:] / 2, np.sqrt(gt[:, 2] * gt[:, 3])
        if self.kind == "paper":
            return self.paper(img, gt, gc, gs, W, H, rng)
        if self.kind == "paper_resized":
            return self.paper_resized(img, gt, gc, gs, W, H, rng)
        return self.seeds(img, gt, gc, gs, W, H, rng, cands)

    def paper(self, img, gt, gc, gs, W, H, rng):
        """Equal numbers of UAV crops (centred on an annotated UAV, small jitter) and background crops (no UAV inside)."""
        xs, ys = [], []
        n = self.per_frame // 2
        for j in rng.integers(len(gt), size=n):
            jit = rng.normal(0, 0.15, 2) * gs[j] if self.train else np.zeros(2)
            xs.append(paper_crop(img, *(gc[j] + jit)))
            ys.append(1.0)
        tries = 0
        while len(xs) < 2 * n and tries < 50 * n:
            tries += 1
            c = np.array([rng.uniform(0, W), rng.uniform(0, H)])
            inside = (np.abs(gc[:, 0] - c[0]) < PAPER_W / 2 + gt[:, 2] / 2) & (np.abs(gc[:, 1] - c[1]) < PAPER_H / 2 + gt[:, 3] / 2)
            if not inside.any():
                xs.append(paper_crop(img, *c))
                ys.append(0.0)
        return self.augment(xs, ys, rng)

    def paper_resized(self, img, gt, gc, gs, W, H, rng):
        """As `paper`, on scale-normalised crops: equal numbers of UAV crops and background crops (no UAV inside)."""
        xs, ys = [], []
        n = self.per_frame // 2
        for j in rng.integers(len(gt), size=n):
            jit = rng.normal(0, 0.15, 2) * gs[j] if self.train else np.zeros(2)
            side = gs[j] * np.exp(rng.normal(0, 0.1)) if self.train else gs[j]
            xs.append(paper_resized_crop(img, *(gc[j] + jit), side))
            ys.append(1.0)
        tries = 0
        while len(xs) < 2 * n and tries < 50 * n:
            tries += 1
            c = np.array([rng.uniform(0, W), rng.uniform(0, H)])
            side = gs[int(rng.integers(len(gt)))] * (np.exp(rng.normal(0, 0.1)) if self.train else 1.0)
            hh = PR_ZOOM * side / 2
            hw = hh * PAPER_W / PAPER_H
            inside = (np.abs(gc[:, 0] - c[0]) < hw + gt[:, 2] / 2) & (np.abs(gc[:, 1] - c[1]) < hh + gt[:, 3] / 2)
            if not inside.any():
                xs.append(paper_resized_crop(img, *c, side))
                ys.append(0.0)
        return self.augment(xs, ys, rng)

    def augment(self, xs, ys, rng):
        x = np.stack(xs)
        if self.train:
            flip = rng.random(len(x)) < 0.5
            x[flip] = x[flip][:, :, :, ::-1]
            x = x * rng.uniform(0.85, 1.15, (len(x), 1, 1, 1)).astype(np.float32) + rng.uniform(-0.15, 0.15, (len(x), 1, 1, 1)).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), torch.tensor(ys, dtype=torch.float32)

    def seeds(self, img, gt, gc, gs, W, H, rng, cands=None):
        """Seeds around annotated UAVs (within and just outside the search radius) and on the background.

        label 1: a UAV centre within R_POS seed sides of the seed centre (target = the nearest one);
        label 0: no UAV centre within R_NEG seed sides; anything between is not used (weight 0).
        """
        xs, sc, lab, tgt = [], [], [], []
        for _ in range(self.per_frame):
            u = rng.random()
            j = int(rng.integers(len(gt)))
            scale = np.exp(rng.normal(0, 0.2)) if self.train else np.exp(rng.normal(0, 0.1))
            sw, sh = gt[j, 2] * scale * np.exp(rng.normal(0, 0.08)), gt[j, 3] * scale * np.exp(rng.normal(0, 0.08))
            side = float(np.sqrt(sw * sh))
            ang = rng.uniform(0, 2 * np.pi)
            if u < 0.5:      # the UAV is inside the search radius
                r = (rng.uniform(0, 0.4) if rng.random() < 0.5 else rng.uniform(0, R_POS)) * side
                c = gc[j] + r * np.array([np.cos(ang), np.sin(ang)])
            elif u < 0.75:   # just outside: the neighbourhood of a UAV, not the UAV
                c = gc[j] + rng.uniform(R_NEG, 3.0) * side * np.array([np.cos(ang), np.sin(ang)])
            else:            # anywhere
                c = np.array([rng.uniform(0, W), rng.uniform(0, H)])
            d = np.linalg.norm(gc - c, axis=1) / side
            k = int(d.argmin())
            y = 1.0 if d[k] <= R_POS else 0.0 if d[k] >= R_NEG else -1.0
            flip = bool(self.train and rng.random() < 0.5)
            x = seed_crop(img, c[0], c[1], side, flip)
            if self.train:
                x = np.clip(x * rng.uniform(0.85, 1.15) + rng.uniform(-0.08, 0.08), 0, 1).astype(np.float32)
            dx = (gc[k, 0] - c[0]) / side
            xs.append(x)
            sc.append(scalars(np.array(side), np.array(float(W))))
            lab.append(y)
            tgt.append([-dx if flip else dx, (gc[k, 1] - c[1]) / side, np.log(gt[k, 2] / sw), np.log(gt[k, 3] / sh)])
        blab = [-1.0] * len(xs)  # Exp112: the birth output is not trained on these seeds
        if self.birth_per_frame and cands is not None and len(cands):
            # detection boxes of this frame as seeds: half positives, half negatives when both exist
            pos, neg = np.flatnonzero(cands[:, 4] > 0.5), np.flatnonzero(cands[:, 4] < 0.5)
            k = self.birth_per_frame
            n_pos = min(len(pos), k // 2)
            n_neg = min(len(neg), k - n_pos)
            n_pos = min(len(pos), k - n_neg)
            pick = np.r_[rng.permutation(pos)[:n_pos], rng.permutation(neg)[:n_neg]]
            for q in pick.astype(int):
                bx = cands[q, :4]
                c, sw, sh = bx[:2] + bx[2:] / 2, float(bx[2]), float(bx[3])
                side = float(np.sqrt(sw * sh))
                flip = bool(self.train and rng.random() < 0.5)
                x = seed_crop(img, c[0], c[1], side, flip)
                if self.train:
                    x = np.clip(x * rng.uniform(0.85, 1.15) + rng.uniform(-0.08, 0.08), 0, 1).astype(np.float32)
                xs.append(x)
                sc.append(scalars(np.array(side), np.array(float(W))))
                y = float(cands[q, 4])
                lab.append(y if y > 0.5 else -1.0)  # the fill-in output only learns the positives' regression from them
                blab.append(y)
                k_ = int(cands[q, 5])
                dx = (gc[k_, 0] - c[0]) / side if y > 0.5 else 0.0
                tgt.append([-dx if flip else dx, (gc[k_, 1] - c[1]) / side if y > 0.5 else 0.0,
                            np.log(gt[k_, 2] / sw) if y > 0.5 else 0.0, np.log(gt[k_, 3] / sh) if y > 0.5 else 0.0])
        return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(sc)), torch.tensor(lab, dtype=torch.float32),
                torch.tensor(np.array(tgt), dtype=torch.float32), torch.tensor(blab, dtype=torch.float32))
