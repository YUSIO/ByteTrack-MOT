"""Exp121: slot-anchored weakening of targets in training images (labels kept), with matched negatives.

Positives: members the real tracker had at t-1 (active slots) whose annotated target is still in frame t; with probability
q each (at most k per frame), the target is made weak in the image of frame t by one of
  blur       Gaussian blur, sigma in [1, 2.5] px;
  downup     shrink by a factor in [0.3, 0.6] and enlarge back (a farther, smaller target);
  contrast   pull towards the local background mean, keeping a share in [0.3, 0.6] of the contrast;
  occlude    cover a share in [0.3, 0.6] of the box, from one side, with a background patch taken 2-3 sides away.
Negatives (as many as positives): the same operations at lost slots whose target is not in the frame (where the tracker has
a prior but nothing is there) and at random background places away from every annotated target, so that neither a
weakened patch nor a prior is evidence of a target by itself. Every operation is blended with a feathered mask.
Ground truth is unchanged. Evaluation stays on real, unmodified frames.
"""
import cv2
import numpy as np

OPS = ("blur", "downup", "contrast", "occlude")


def _feather(h, w):
    y = np.minimum(np.arange(h) + 0.5, h - np.arange(h) - 0.5) / max(h / 2, 1)
    x = np.minimum(np.arange(w) + 0.5, w - np.arange(w) - 0.5) / max(w / 2, 1)
    return np.clip(np.minimum(y[:, None], x[None, :]) * 2.5, 0, 1)[..., None]


def _region(cx, cy, bw, bh, W, H, grow=1.6):
    w, h = max(bw * grow, 6), max(bh * grow, 6)
    x0, y0 = int(max(0, np.floor(cx - w / 2))), int(max(0, np.floor(cy - h / 2)))
    x1, y1 = int(min(W, np.ceil(cx + w / 2))), int(min(H, np.ceil(cy + h / 2)))
    return x0, y0, x1, y1


def apply_op(img, box, op, rng, gt=None):
    """img: H x W x 3 uint8 (modified in place); box: (cx, cy, w, h) in pixels."""
    H, W = img.shape[:2]
    cx, cy, bw, bh = box
    x0, y0, x1, y1 = _region(cx, cy, bw, bh, W, H)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return False
    crop = img[y0:y1, x0:x1].astype(np.float32)
    if op == "blur":
        new = cv2.GaussianBlur(crop, (0, 0), float(rng.uniform(1.0, 2.5)))
    elif op == "downup":
        f = float(rng.uniform(0.3, 0.6))
        small = cv2.resize(crop, (max(1, int(round(crop.shape[1] * f))), max(1, int(round(crop.shape[0] * f)))), interpolation=cv2.INTER_AREA)
        new = cv2.resize(small, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
    elif op == "contrast":
        ring = np.concatenate([crop[:2].reshape(-1, 3), crop[-2:].reshape(-1, 3), crop[:, :2].reshape(-1, 3), crop[:, -2:].reshape(-1, 3)])
        mu = ring.mean(0)
        new = mu + float(rng.uniform(0.3, 0.6)) * (crop - mu)
    else:  # occlude: a background patch from 2-3 sides away covers part of the box, from one side
        side = float(np.sqrt(max(bw * bh, 1.0)))
        for _ in range(8):
            ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(2.0, 3.0) * side
            sx, sy = int(round(x0 + np.cos(ang) * dist)), int(round(y0 + np.sin(ang) * dist))
            if 0 <= sx and 0 <= sy and sx + (x1 - x0) <= W and sy + (y1 - y0) <= H:
                if gt is None or not len(gt) or np.all(np.hypot(gt[:, 0] - (sx + (x1 - x0) / 2), gt[:, 1] - (sy + (y1 - y0) / 2)) > 1.5 * side):
                    break
        else:
            return False
        patch = img[sy:sy + (y1 - y0), sx:sx + (x1 - x0)].astype(np.float32)
        share = float(rng.uniform(0.3, 0.6))
        m = np.zeros(crop.shape[:2], np.float32)
        hh, ww = m.shape
        side_i = int(rng.integers(4))
        if side_i == 0:
            m[:, :max(1, int(ww * share))] = 1
        elif side_i == 1:
            m[:, ww - max(1, int(ww * share)):] = 1
        elif side_i == 2:
            m[:max(1, int(hh * share)), :] = 1
        else:
            m[hh - max(1, int(hh * share)):, :] = 1
        m = cv2.GaussianBlur(m, (0, 0), 1.0)[..., None]
        new = crop * (1 - m) + patch * m
    a = _feather(*crop.shape[:2])
    img[y0:y1, x0:x1] = np.clip(crop * (1 - a) + new * a, 0, 255).astype(np.uint8)
    return True


def weaken(img, slots, gt_tlwh, cfg, rng, order):
    """Weaken targets of one frame in place. slots: the frame's slots (fctrack.slot_prior), gt_tlwh: n x 4 annotated boxes.
    order: the lost-slot predictor order of the prior (to place negatives at lost slots). Returns counts per kind."""
    from fctrack.slot_prior import pick
    stats = {"pos": 0, "neg_lost": 0, "neg_bg": 0}
    if rng.random() >= cfg.get("p_frame", 0.5):
        return stats
    H, W = img.shape[:2]
    g = np.asarray(gt_tlwh, float).reshape(-1, 4)
    gc = np.c_[g[:, 0] + g[:, 2] / 2, g[:, 1] + g[:, 3] / 2, g[:, 2], g[:, 3]]
    pos = []
    for sl in slots:
        if sl["kind"] != 0 or not len(gc):
            continue
        cx, cy, _ = pick(sl, order)
        d = np.hypot(gc[:, 0] - cx, gc[:, 1] - cy) / np.sqrt(np.maximum(gc[:, 2] * gc[:, 3], 1))
        j = int(d.argmin())
        if d[j] <= 0.5 and rng.random() < cfg.get("q", 0.3):
            pos.append(j)
    pos = list(dict.fromkeys(pos))[:cfg.get("max_per_frame", 5)]
    for j in pos:
        if apply_op(img, gc[j], OPS[int(rng.integers(len(OPS)))], rng, gc):
            stats["pos"] += 1
    n_neg = stats["pos"]
    lost = []
    for sl in slots:
        if sl["kind"] != 1:
            continue
        cx, cy, _ = pick(sl, order)
        if 0 <= cx < W and 0 <= cy < H and (not len(gc) or np.min(np.hypot(gc[:, 0] - cx, gc[:, 1] - cy)) > 1.5 * sl["side"]):
            lost.append((cx, cy, sl["side"], sl["side"]))
    rng.shuffle(lost)
    sides = np.sqrt(np.maximum(gc[:, 2] * gc[:, 3], 1)) if len(gc) else np.array([15.0])
    for b in lost[:n_neg // 2 + n_neg % 2]:
        if apply_op(img, b, OPS[int(rng.integers(len(OPS)))], rng, gc):
            stats["neg_lost"] += 1
    for _ in range(40):
        if stats["neg_lost"] + stats["neg_bg"] >= n_neg:
            break
        s = float(rng.choice(sides))
        if W - s <= s or H - s <= s:  # image smaller than two sides: no background place (Exp121 failed here)
            continue
        cx, cy = rng.uniform(s, W - s), rng.uniform(s, H - s)
        if len(gc) and np.min(np.hypot(gc[:, 0] - cx, gc[:, 1] - cy)) <= 2.0 * s:
            continue
        if apply_op(img, (cx, cy, s, s), OPS[int(rng.integers(len(OPS)))], rng, gc):
            stats["neg_bg"] += 1
    return stats
