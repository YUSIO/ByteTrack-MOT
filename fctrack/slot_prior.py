"""Exp120: member slots of the tracker as a prior heatmap for the detector (CenterTrack-style input, rendered from slots).

The slots of frame t are computed from the rows the tracker wrote up to frame t-1 only (online; tracker-agnostic, nothing
inside ByteTrack is read):

  kind 0  active   a member written at t-1; its centre is carried to t with its own constant velocity (`cv`).
  kind 1  lost     a member last written at f1 with 2 <= t - f1 <= max_gap; its centre at t is predicted by several
                   estimators, all kept so that they can be compared and chosen without re-running the tracker:
                     hold      its last centre;
                     cv        its last centre plus its own velocity times the gap;
                     mst       each neighbour on the minimum spanning tree of the members written at f1 keeps its offset to
                               the member (fctrack.formation.estimate, model offset); the neighbours' centres at t are their
                               own constant-velocity carries from t-1;
                     delaunay  the same with the Delaunay natural neighbours;
                     affine    one affine map fitted to the Delaunay neighbours (barycentric inside their triangle).
                   Every graph estimate also returns its spread (the largest distance between two neighbour estimates).

The heatmap has one channel per kind. Each slot is a Gaussian peak (max over slots, CenterTrack's render function) with
a standard deviation in original pixels
  active  sigma = a0 * side
  lost    sigma = sqrt((a1 * side)^2 + (b1 * spread)^2 + c1^2 * side^2 * gap)
evaluated at the centres of the stride-s cells of the letterboxed input, widened by half a cell (no aliasing of peaks
smaller than a cell).
"""
import numpy as np

from fctrack.formation import estimate, select_neighbours

PREDICTORS = ("hold", "cv", "mst", "delaunay", "affine")


class SlotPrior:
    def __init__(self, cfg=None):
        c = cfg or {}
        self.max_gap = int(c.get("max_gap", 30))
        self.vel_frames = int(c.get("vel_frames", 5))
        self.max_neighbours = int(c.get("max_neighbours", 6))
        self.consistency = float(c.get("consistency_sides", 0.5))
        self.begin_sequence()

    def begin_sequence(self):
        self.hist = {}  # frame -> {id: (cx, cy, side)} of the rows written in that frame
        self.last = {}  # id -> last frame it was written

    def observe(self, f, rows):
        """rows: iterable of (track id, tlwh) written in frame f (boxes with zero area are ignored)."""
        h = {}
        for tid, b in rows:
            b = np.asarray(b, float)
            if b[2] * b[3] > 0:
                h[int(tid)] = (b[0] + b[2] / 2, b[1] + b[3] / 2, float(np.sqrt(b[2] * b[3])))
                self.last[int(tid)] = f
        self.hist[f] = h
        for old in [x for x in self.hist if x < f - self.max_gap - self.vel_frames - 1]:
            del self.hist[old]
        for tid in [k for k, v in self.last.items() if v < f - self.max_gap]:
            del self.last[tid]

    def velocity(self, tid, f_end):
        c1 = self.hist[f_end][tid]
        for k in range(self.vel_frames, 0, -1):
            c0 = self.hist.get(f_end - k, {}).get(tid)
            if c0 is not None:
                return np.array([(c1[0] - c0[0]) / k, (c1[1] - c0[1]) / k])
        return np.zeros(2)

    def slots(self, t):
        """Slots of frame t: list of dicts {kind, id, gap, side, pred: {name: (cx, cy, spread)}}."""
        prev = self.hist.get(t - 1, {})
        out, act = [], {}
        for tid, (cx, cy, side) in prev.items():
            p = np.array([cx, cy]) + self.velocity(tid, t - 1)
            act[tid] = p
            out.append({"kind": 0, "id": tid, "gap": 1, "side": side, "pred": {"cv": (float(p[0]), float(p[1]), 0.0)}})
        for tid, f1 in self.last.items():
            gap = t - f1
            if gap < 2 or gap > self.max_gap or f1 not in self.hist or tid not in self.hist[f1]:
                continue
            cx, cy, side = self.hist[f1][tid]
            c1 = np.array([cx, cy])
            v = self.velocity(tid, f1)
            pred = {"hold": (cx, cy, 0.0)}
            q = c1 + v * gap
            pred["cv"] = (float(q[0]), float(q[1]), 0.0)
            ref = self.hist[f1]
            ids = list(ref)
            m = ids.index(tid)
            P0 = np.array([ref[i][:2] for i in ids])
            avail = np.array([i in act and i != tid for i in ids])
            eps = self.consistency * side
            if avail.any():
                for name, graph, model in (("mst", "mst", "offset"), ("delaunay", "delaunay", "offset"), ("affine", "delaunay", "affine")):
                    nb = select_neighbours(P0, m, avail, graph, self.max_neighbours)
                    if len(nb):
                        N1 = np.array([act[ids[j]] for j in nb])
                        c, spread, _ = estimate(P0[m], P0[nb], N1, model, eps)
                        pred[name] = (float(c[0]), float(c[1]), float(spread))
            out.append({"kind": 1, "id": tid, "gap": gap, "side": side, "pred": pred})
        return out


def pick(slot, order):
    """(cx, cy, spread) of the first predictor in `order` the slot has (active slots always use cv)."""
    if slot["kind"] == 0:
        return slot["pred"]["cv"]
    for name in order:
        if name in slot["pred"]:
            return slot["pred"][name]
    return slot["pred"]["cv"]


def sigma(slot, spread, s):
    side = slot["side"]
    if slot["kind"] == 0:
        return s["a0"] * side
    return float(np.sqrt((s["a1"] * side) ** 2 + (s["b1"] * spread) ** 2 + s["c1"] ** 2 * side ** 2 * slot["gap"]))


def render(slots, cfg, scale, input_hw, stride=8, flip_width=None):
    """2 x (H/stride) x (W/stride) float32 heatmap in the letterboxed input (YOLOX preproc: scale, no offset).

    cfg: {"lost_order": [...], "sigma": {"a0", "a1", "b1", "c1"}}; flip_width: original image width when the frame is
    flipped horizontally (training augmentation)."""
    H, W = input_hw[0] // stride, input_hw[1] // stride
    out = np.zeros((2, H, W), np.float32)
    if not slots:
        return out
    ys = (np.arange(H) + 0.5) * stride
    xs = (np.arange(W) + 0.5) * stride
    for sl in slots:
        cx, cy, spread = pick(sl, cfg["lost_order"])
        if flip_width is not None:
            cx = flip_width - cx
        sg = sigma(sl, spread, cfg["sigma"]) * scale
        e = float(np.sqrt(sg ** 2 + (stride / 2) ** 2))
        px, py = cx * scale, cy * scale
        x0, x1 = np.searchsorted(xs, px - 3 * e), np.searchsorted(xs, px + 3 * e)
        y0, y1 = np.searchsorted(ys, py - 3 * e), np.searchsorted(ys, py + 3 * e)
        if x1 <= x0 or y1 <= y0:
            continue
        g = np.exp(-((xs[None, x0:x1] - px) ** 2 + (ys[y0:y1, None] - py) ** 2) / (2 * e * e))
        k = sl["kind"]
        np.maximum(out[k, y0:y1, x0:x1], g, out=out[k, y0:y1, x0:x1])
    return out
