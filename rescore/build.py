"""Exp094: candidate table for temporal re-scoring of a frozen detection cache.

For every detection with score >= `min_score` the table holds multi-frame image crops (frames t, t-1, t-2 at the box
position, plus one wide crop of frame t), scalar cues computed from the detection cache alone, and, when a ground truth
root is given, the label: the box is the best-IoU box (IoU >= 0.5) of some annotated row in its frame. Ground truth is read
only for the label; nothing is read from any tracker output. Usage:
  build.py --data <root with <seq>/img1> --dets <cache root> --out <npz> [--gt <root with <seq>/gt/gt.txt>] [--seqs ...]
"""
import argparse
import configparser
from pathlib import Path

import cv2
import numpy as np

CROP = 64
SCALARS = ("score", "log_side", "log_aspect", "cx", "cy", "prev1", "prev2", "n_strong_near", "dist_strong", "dup_iou", "n_higher_overlap", "log_n_strong")


def iou(a, b):
    """a: n x 4, b: m x 4, both x, y, w, h."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0][:, None], b[:, 0][None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1][:, None], b[:, 1][None]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:3] * a[:, 3:4] + (b[:, 2] * b[:, 3])[None] - inter + 1e-9)


def crop(img, cx, cy, window):
    s = CROP / window
    M = np.array([[s, 0, CROP / 2 - s * cx], [0, s, CROP / 2 - s * cy]], np.float32)
    return cv2.warpAffine(img, M, (CROP, CROP), flags=cv2.INTER_LINEAR if s >= 1 else cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def sequence(data, dets, gt, min_score):
    ini = configparser.ConfigParser()
    ini.read(data / "seqinfo.ini")
    L, W, H = int(ini["Sequence"]["seqlength"]), int(ini["Sequence"]["imwidth"]), int(ini["Sequence"]["imheight"])
    raw = np.loadtxt(dets / "det.txt", delimiter=",", ndmin=2)
    raw = raw[raw[:, 6] >= min_score]
    per = {f: raw[raw[:, 0] == f][:, [2, 3, 4, 5, 6]] for f in range(1, L + 1)}  # x, y, w, h, score
    gtf = None
    if gt is not None:
        g = np.loadtxt(gt, delimiter=",", ndmin=2)
        gtf = {f: g[g[:, 0] == f][:, 2:6] for f in range(1, L + 1)}
    frames = {}

    def image(f):
        f = min(max(f, 1), L)
        if f not in frames:
            frames[f] = cv2.imread(str(data / "img1" / f"{f:06d}.jpg"))
            for old in [k for k in frames if k < f - 3]:
                del frames[old]
        return frames[f]

    out = {"crops": [], "scal": [], "label": [], "iou_gt": [], "frame": [], "box": []}
    for f in range(1, L + 1):
        b = per[f]
        if len(b) == 0:
            continue
        boxes, score = b[:, :4], b[:, 4]
        side = np.sqrt(np.maximum(boxes[:, 2] * boxes[:, 3], 4.0))
        ctr = boxes[:, :2] + boxes[:, 2:] / 2
        m = iou(boxes, boxes)
        np.fill_diagonal(m, 0)
        higher = score[None, :] > score[:, None]
        dup = (m * higher).max(1)
        n_higher = ((m > 0.5) & higher).sum(1)
        dist = np.linalg.norm(ctr[:, None] - ctr[None], axis=2) / side[:, None]
        strong = (score >= 0.7)[None, :] & (m < 0.3)
        d_strong = np.where(strong, dist, 99.0).min(1) if len(b) > 1 else np.full(len(b), 99.0)
        n_near = (((score >= 0.6)[None, :]) & (dist <= 6) & (m < 0.3)).sum(1)
        prev = []
        for k in (1, 2):
            p = per.get(f - k, np.zeros((0, 5)))
            mm = iou(boxes, p[:, :4]) if len(p) else np.zeros((len(b), 0))
            prev.append(np.where(mm.max(1) >= 0.3, p[mm.argmax(1), 4], 0.0) if mm.shape[1] else np.zeros(len(b)))
        if gtf is not None:
            g = gtf[f]
            mg = iou(boxes, g)
            label = np.zeros(len(b))
            if len(g):
                for j in range(len(g)):
                    i = mg[:, j].argmax()
                    if mg[i, j] >= 0.5:
                        label[i] = 1.0
            iou_gt = mg.max(1) if len(g) else np.zeros(len(b))
        else:
            label, iou_gt = np.full(len(b), -1.0), np.full(len(b), -1.0)
        im0, im1, im2 = image(f), image(f - 1), image(f - 2)
        for i in range(len(b)):
            cx, cy = ctr[i]
            out["crops"].append(np.concatenate([crop(im0, cx, cy, 3 * side[i]), crop(im1, cx, cy, 3 * side[i]), crop(im2, cx, cy, 3 * side[i]),
                                                crop(im0, cx, cy, 8 * side[i])], axis=2).transpose(2, 0, 1))
            out["scal"].append([score[i], np.log(side[i] / 20.0), np.log(boxes[i, 2] / max(boxes[i, 3], 1e-3)), cx / W, cy / H, prev[0][i], prev[1][i],
                                min(n_near[i], 20), min(d_strong[i], 30.0) / 10.0, dup[i], min(n_higher[i], 5), np.log1p((score >= 0.6).sum())])
        out["label"].extend(label); out["iou_gt"].extend(iou_gt); out["frame"].extend([f] * len(b)); out["box"].extend(boxes.tolist())
    return {k: np.array(v) for k, v in out.items()}, (W, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--dets", type=Path, required=True)
    ap.add_argument("--gt", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seqs", nargs="*")
    ap.add_argument("--min-score", type=float, default=0.1)
    a = ap.parse_args()
    seqs = a.seqs or sorted(p.name for p in a.dets.glob("UAVSwarm-*") if p.is_dir())
    parts, seq_id = [], []
    for s in seqs:
        gt = (a.gt / s / "gt/gt.txt") if a.gt else None
        t, _ = sequence(a.data / s, a.dets / s, gt, a.min_score)
        parts.append(t); seq_id.extend([int(s[-2:])] * len(t["frame"]))
        print(s, len(t["frame"]), int((t["label"] > 0.5).sum()), flush=True)
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez_compressed(a.out, seq=np.array(seq_id), scalar_names=np.array(SCALARS), **{k: (v.astype(np.uint8) if k == "crops" else v.astype(np.float32)) for k, v in cat.items()})
    print("rows", len(cat["frame"]), "positives", int((cat["label"] > 0.5).sum()))


if __name__ == "__main__":
    main()
