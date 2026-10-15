"""Inputs of one track query, built the same way from annotations (training) and from the tracker (inference).

Everything is computed in original-image pixels and converted to network-input pixels (x r) at the end, with an
optional horizontal flip of the original image (training augmentation).
"""
import numpy as np
import torch

from jdt.head import MAX_NB, MAX_OCC

MIN_HALF = 24.0  # network-input pixels: the window is never smaller than 6 x 6 stride-8 cells


def geometry(cx, cy, w, h, W, H, member_sides):
    side = float(np.sqrt(w * h))
    edge = min(cx, cy, W - cx, H - cy) / side
    med = float(np.median(member_sides)) if len(member_sides) else side
    return [np.log(side / 20.0), np.log(w / h), cx / W, cy / H, min(max(edge, 0.0), 10.0) / 10.0, np.log(med / side)]


def make_query(*, centre, box, tbox, cv, vel, gap, nb, spread, occ, member_sides, W, H, r, flip=False, gt=None):
    """centre: placement (x, y); box: last detector box of the track (x1, y1, w, h); tbox: template box (x1, y1, w, h)
    in the template frame; cv: constant-velocity centre; vel: (vx, vy) per frame; nb: k x 2 neighbour estimates;
    spread: largest distance between two neighbour estimates in pixels; occ: m x 2 centres of the other members tracked
    in this frame; gt: (cx, cy, w, h) of the track's UAV in this frame or None. Original-image pixels."""
    w, h = float(box[2]), float(box[3])
    side = float(np.sqrt(w * h))
    geo = geometry(box[0] + w / 2, box[1] + h / 2, w, h, W, H, member_sides)
    fx = (lambda x: W - x) if flip else (lambda x: x)
    pt = lambda p: np.array([fx(p[0]), p[1]], np.float64) * r
    tw, th = float(tbox[2]), float(tbox[3])
    tc = np.array([tbox[0] + tw / 2, tbox[1] + th / 2])
    ts = max(1.5 * np.sqrt(tw * th) * r, 16.0)  # template roi: 1.5 x its side, at least two stride-8 cells
    tcn = pt(tc)
    if flip:
        geo[2] = 1.0 - geo[2]
    q = {
        "centre": pt(centre), "half": max(1.5 * side * r, MIN_HALF), "side": side * r, "wh": np.array([w, h]) * r,
        "tbox": np.r_[tcn - ts / 2, tcn + ts / 2], "cv": pt(cv), "vel": np.array([-vel[0] if flip else vel[0], vel[1]]) * r,
        "gap": float(gap), "spread": float(spread) / side, "geo": np.array(geo, np.float64),
        "nb": np.zeros((MAX_NB, 2)), "nb_mask": np.zeros(MAX_NB), "occ": np.zeros((MAX_OCC, 2)), "occ_mask": np.zeros(MAX_OCC),
    }
    for i, p in enumerate(np.asarray(nb, float).reshape(-1, 2)[:MAX_NB]):
        q["nb"][i], q["nb_mask"][i] = pt(p), 1.0
    occ = np.asarray(occ, float).reshape(-1, 2)
    if len(occ):  # the nearest ones to the placement
        occ = occ[np.argsort(np.linalg.norm(occ - np.asarray(centre, float), axis=1))[:MAX_OCC]]
    for i, p in enumerate(occ):
        q["occ"][i], q["occ_mask"][i] = pt(p), 1.0
    if gt is not None:
        g = pt(gt[:2])
        q["gt"] = np.r_[g, np.array(gt[2:4], float) * r]
        q["exist"] = float(abs(g[0] - q["centre"][0]) <= q["half"] and abs(g[1] - q["centre"][1]) <= q["half"])
    else:
        q["gt"], q["exist"] = np.r_[q["centre"], q["wh"]], 0.0
    return q


KEYS = ("centre", "half", "side", "wh", "tbox", "cv", "vel", "gap", "spread", "geo", "nb", "nb_mask", "occ", "occ_mask", "gt", "exist")


def stack(queries, device):
    """List of query dicts -> dict of float32 tensors."""
    return {k: torch.as_tensor(np.stack([np.asarray(q[k], np.float32) for q in queries]), device=device) for k in KEYS}
