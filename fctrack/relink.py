"""Online relabelling at birth (Exp076): when a new track id appears, decide in that frame whether it is a member
whose track was lost, and if so write it under the old label from then on.

The pass reads a tracker's output (MOT rows) frame by frame and uses only rows of the current and earlier frames, so
it is causal: no row is ever rewritten after it was written. Boxes and frames are unchanged; only labels change.

Per new id n (first row in frame f0 > first frame of the sequence) the candidates are the labels whose holder was last
written in frame f1 with 1 <= f0 - f1 <= G and is not written in f0. Each candidate's last centre c1 is carried to f0:

  abs    not moved
  gmc    through the camera-motion warps of frames f1+1 .. f0 (background estimate, Exp055)
  seat   by the median displacement of the peers written in both f1 and f0 (Exp068 `seat`)
  form   by the FCTrack formation estimate: Delaunay neighbours among the members written in f1 that are written in
         f0, offset model (fctrack.formation.estimate, the Exp070 setting)

`seat` and `form` fall back to `abs` when no peer is available. The distance is |carried c1 - c0| over the side
(sqrt(w h)) of n's first box. New ids of one frame are assigned greedily by distance; the nearest candidate within tau
gives its label. A holder that is written again after its label was taken gets a fresh label.
"""
import numpy as np

from fctrack.formation import estimate, select_neighbours

FAMILIES = ("abs", "gmc", "seat", "form")


def centre(r):
    return np.array([r[2] + r[4] / 2, r[3] + r[5] / 2])


def carry(family, c1, side1, f1, f0, label, frames, warps):
    """Centre of the candidate in frame f0; frames[f] maps label -> centre of the rows written in frame f."""
    if family == "abs":
        return c1
    if family == "gmc":
        p = c1.copy()
        for f in range(f1 + 1, f0 + 1):
            w = warps.get(f)
            if w is not None:
                p = w[:, :2] @ p + w[:, 2]
        return p
    a, b = frames.get(f1, {}), frames.get(f0, {})
    peers = [k for k in a if k in b and k != label]
    if not peers:
        return c1
    if family == "seat":
        return c1 + np.median([b[k] - a[k] for k in peers], axis=0)
    ids = list(a)
    P0 = np.array([a[k] for k in ids])
    avail = np.array([k in b and k != label for k in ids])
    nb = select_neighbours(P0, ids.index(label), avail, "delaunay", 6)
    if not nb:
        return c1
    N0, N1 = P0[nb], np.array([b[ids[j]] for j in nb])
    return estimate(c1, N0, N1, "offset", 0.5 * side1)[0]  # consistency 0.5 sides of the member's box, as in Exp070


def relabel(rows, family, tau, gap, warps=None, log=None):
    """rows: n x >=7 array (frame, id, x, y, w, h, score, ...). Returns a copy with relabelled ids."""
    warps = warps or {}
    out = rows.copy()
    if not len(rows):
        return out
    order = np.lexsort((rows[:, 1], rows[:, 0]))
    rows, out = rows[order], out[order]
    first_frame = int(rows[0, 0])
    label = {}            # original id -> label written
    holder = {}           # label -> original id currently holding it
    last = {}             # label -> (frame, centre, side) of its last written row
    frames = {}           # frame -> {label: centre}
    nxt = int(rows[:, 1].max()) + 1
    seen = set()
    for f in np.unique(rows[:, 0]).astype(int):
        idx = np.where(rows[:, 0] == f)[0]
        ids = [int(rows[i, 1]) for i in idx]
        births = [i for i, t in zip(idx, ids) if t not in seen and f > first_frame]
        for t in ids:  # holders that lost their label to a claimant come back under a fresh label
            if t in seen and holder.get(label.get(t)) != t:
                label[t], nxt = nxt, nxt + 1
                holder[label[t]] = t
        present = {label[t] for t in ids if t in label}
        cur = {label[int(rows[i, 1])]: centre(rows[i]) for i in idx if int(rows[i, 1]) in label}
        frames[f] = cur  # members already written in f (births are added below)
        pairs = []
        for i in births:
            c0, side = centre(rows[i]), float(np.sqrt(max(rows[i, 4] * rows[i, 5], 1.0)))
            for L, (f1, c1, side1) in last.items():
                if L in present or not 1 <= f - f1 <= gap:
                    continue
                d = float(np.linalg.norm(carry(family, c1, side1, f1, f, L, frames, warps) - c0)) / side
                if d <= tau:
                    pairs.append((d, i, L, f - f1))
        taken_i, taken_L = set(), set()
        for d, i, L, g in sorted(pairs):
            if i in taken_i or L in taken_L:
                continue
            t = int(rows[i, 1])
            label[t] = L
            holder[L] = t
            taken_i.add(i); taken_L.add(L)
            if log is not None:
                log.append((f, t, L, g, d))
        for i in births:
            t = int(rows[i, 1])
            seen.add(t)
            if t not in label:  # a new id that was not relabelled keeps its own id (original ids never collide)
                label[t] = t
                holder[t] = t
        for i in idx:
            t = int(rows[i, 1])
            if t not in label:  # ids of the first frame
                label[t] = holder[t] = t
            seen.add(t)
            L = label[t]
            out[i, 1] = L
            frames[f][L] = centre(rows[i])
            last[L] = (f, centre(rows[i]), float(np.sqrt(max(rows[i, 4] * rows[i, 5], 1.0))))
    return out[np.argsort(order)]
