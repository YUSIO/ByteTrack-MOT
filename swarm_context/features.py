"""Explicit swarm-state features for every box of a detection cache (Exp066). Causal, no images, no tracker.

Boxes are chained from frame to frame (a box takes the nearest unclaimed box of one of the three previous frames that
overlaps it or lies within one box side per frame). A chain with at least three earlier boxes whose mean score is at
least 0.6 is an established member. Features of a box:

  own (O)    its score, and what its own chain says: length, past scores, displacement, size change, duplicates
  swarm (G)  the state of the established members other than its own chain in the same frame, and how the box relates
             to it: how many there are and how many have just gone missing, how far their scores are from their own past
             levels (collective confidence offset), the box's size and motion relative to theirs, how far it is from the
             nearest one and from where a missing one would be now, and how many stray weak boxes the frame has
"""
import numpy as np

from swarm_context.data import iou_matrix

OWN = ["logit", "has_prev", "gap", "length", "mean_past", "max_past", "trend", "frac_high", "disp", "dsize", "iou_prev", "aspect", "border", "dup", "dup_higher"]
SWARM = ["n_members", "n_missing", "frac_missing", "conf_offset", "frac_drop", "rel_logit", "rel_size", "rel_size_local", "rel_motion", "has_motion",
         "near_member", "near_missing", "clutter", "n_members_change"]
GROUPS = {"count": ["n_members", "n_missing", "frac_missing", "n_members_change"], "confidence": ["conf_offset", "frac_drop", "rel_logit"],
          "size": ["rel_size", "rel_size_local"], "motion": ["rel_motion", "has_motion"], "place": ["near_member", "near_missing"], "clutter": ["clutter"]}


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def extract(seq, k=8, memory=30):
    """seq: swarm_context.data.Sequence. Returns dict of arrays over all boxes of frames first..last, in frame order."""
    rows = []
    prev = {}  # frame -> list of per-box state dicts
    missing = []  # established chains without a box: dicts with position, side, frames gone
    n_hist = []
    for t in range(seq.first, seq.last + 1):
        det = seq.det[t]
        n = len(det)
        c = det[:, :2] + det[:, 2:4] / 2 if n else np.zeros((0, 2))
        side = np.sqrt(det[:, 2] * det[:, 3]) if n else np.zeros(0)
        lg = logit(det[:, 4]) if n else np.zeros(0)
        state = [None] * n
        claimed = {g: set() for g in (1, 2, 3)}
        ious = iou_matrix(det[:, :4], det[:, :4]) if n else np.zeros((0, 0))
        for i in np.argsort(-det[:, 4]) if n else []:
            best = None
            for g in (1, 2, 3):
                cand = prev.get(t - g)
                if not cand:
                    continue
                pb = np.array([s["box"] for s in cand])
                pc = pb[:, :2] + pb[:, 2:4] / 2
                io = iou_matrix(det[i:i + 1, :4], pb[:, :4])[0]
                dist = np.linalg.norm(pc - c[i], axis=1) / side[i]
                ratio = np.sqrt(pb[:, 2] * pb[:, 3]) / side[i]
                ok = ((io >= 0.2) | (dist <= 1.0 * g)) & (ratio > 0.5) & (ratio < 2.0)
                ok &= ~np.isin(np.arange(len(cand)), list(claimed[g]))
                if ok.any():
                    j = int(np.lexsort((dist, -io))[np.flatnonzero(ok[np.lexsort((dist, -io))])[0]])
                    best = (g, j, float(io[j]))
                    break
            if best is None:
                hist, disp, dsz, iop, gap, vel = [], 0.0, 0.0, 0.0, 4, None
            else:
                g, j, iop = best
                claimed[g].add(j)
                p = prev[t - g][j]
                hist = (p["hist"] + [p["lg"]])[-k:]
                pcen = p["box"][:2] + p["box"][2:4] / 2
                vel = (c[i] - pcen) / g
                disp = float(np.linalg.norm(vel) / side[i])
                dsz = float(abs(np.log(det[i, 2] * det[i, 3] / (p["box"][2] * p["box"][3]))))
                gap = g
                p["continued"] = True
            est = len(hist) >= 3 and float(np.mean([1 / (1 + np.exp(-h)) for h in hist])) >= 0.6
            state[i] = {"box": det[i].copy(), "lg": float(lg[i]), "hist": hist, "vel": vel, "est": est, "continued": False,
                        "own": [lg[i], float(best is not None), gap, len(hist), np.mean(hist) if hist else lg[i], max(hist) if hist else lg[i], lg[i] - (np.mean(hist) if hist else lg[i]),
                                np.mean([h > 0.405 for h in hist]) if hist else 0.0, disp, dsz, iop, float(np.log(det[i, 2] / det[i, 3])),
                                float(np.arcsinh(max(-2.0, min(c[i, 0], seq.width - c[i, 0], c[i, 1], seq.height - c[i, 1]) / side[i]))),
                                float((ious[i] > 0.3).sum() - 1), float(((ious[i] > 0.5) & (det[:, 4] > det[i, 4])).any())]}
        # established members that had a box in the previous frame and have none now go missing; they are remembered for a while
        est_idx = [i for i in range(n) if state[i]["est"]]
        moved = np.array([state[i]["vel"] for i in est_idx if state[i]["vel"] is not None]).reshape(-1, 2)
        swarm_vel = np.median(moved, axis=0) if len(moved) >= 1 else np.zeros(2)
        for m in missing:
            m["pos"] = m["pos"] + swarm_vel
            m["gone"] += 1
        gone_now = [s for s in prev.get(t - 1, []) if s["est"] and not s["continued"]] + [s for s in prev.get(t - 1, []) if (not s["est"]) and len(s["hist"]) + 1 >= 3 and
                                                                                               np.mean([1 / (1 + np.exp(-h)) for h in s["hist"] + [s["lg"]]]) >= 0.6 and not s["continued"]]
        missing = [m for m in missing if m["gone"] <= memory] + [{"pos": s["box"][:2] + s["box"][2:4] / 2 + swarm_vel, "side": float(np.sqrt(s["box"][2] * s["box"][3])), "gone": 1} for s in gone_now]
        # a missing member that some box has re-found (within one side) is dropped from the list after this frame
        n_est = len(est_idx)
        n_hist.append(n_est)
        base_n = np.median(n_hist[-k - 1:-1]) if len(n_hist) > 1 else n_est
        clutter = sum(1 for i in range(n) if not state[i]["est"] and det[i, 4] < 0.3)
        for i in range(n):
            others = [j for j in est_idx if j != i]
            off = [state[j]["lg"] - np.mean(state[j]["hist"]) for j in others]
            if others:
                oc = c[others]
                d = np.linalg.norm(oc - c[i], axis=1) / side[i]
                near = np.argsort(d)[:3]
                rel_size, rel_local = float(np.log(side[i]) - np.median(np.log(side[others]))), float(np.log(side[i]) - np.mean(np.log(side[others])[near]))
                near_member = float(np.arcsinh(d.min()))
                rel_logit = float(lg[i] - np.median(lg[others]))
            else:
                rel_size = rel_local = rel_logit = 0.0
                near_member = float(np.arcsinh(50.0))
            mv = [state[j]["vel"] for j in others if state[j]["vel"] is not None]
            if state[i]["vel"] is not None and len(mv) >= 1:
                rel_motion, has_motion = float(np.arcsinh(np.linalg.norm(state[i]["vel"] - np.median(np.array(mv), axis=0)) / side[i])), 1.0
            else:
                rel_motion, has_motion = 0.0, 0.0
            if missing:
                dm = min(np.linalg.norm(m["pos"] - c[i]) / max(side[i], m["side"]) for m in missing)
                near_missing = float(np.arcsinh(dm))
            else:
                near_missing = float(np.arcsinh(50.0))
            g = [min(len(others), 20) / 10, min(len(missing), 10) / 5, len(missing) / max(1, len(missing) + len(others)), float(np.median(off)) if off else 0.0,
                 float(np.mean([state[j]["lg"] < 0.405 for j in others])) if others else 0.0, rel_logit, rel_size, rel_local, rel_motion, has_motion, near_member, near_missing,
                 min(clutter, 30) / 10, float(np.clip(n_est - base_n, -5, 5)) / 3]
            rows.append((t, i, state[i]["own"], g))
        # re-found: drop missing members that have an established or strong box within one side now
        if missing and n:
            keep = []
            for m in missing:
                d = np.linalg.norm(c - m["pos"], axis=1) / np.maximum(side, m["side"])
                if not ((d <= 1.0) & (det[:, 4] >= 0.6)).any():
                    keep.append(m)
            missing = keep
        prev[t] = state
        prev.pop(t - 4, None)
    frame = np.array([r[0] for r in rows], np.int64)
    own = np.array([r[2] for r in rows], np.float32).reshape(-1, len(OWN))
    swarm = np.array([r[3] for r in rows], np.float32).reshape(-1, len(SWARM))
    lab = np.concatenate([seq.lab[t] for t in range(seq.first, seq.last + 1)]) if rows else np.zeros(0, np.int64)
    score = np.concatenate([seq.det[t][:, 4] for t in range(seq.first, seq.last + 1)]) if rows else np.zeros(0, np.float32)
    box = np.concatenate([seq.det[t][:, :4] for t in range(seq.first, seq.last + 1)]) if rows else np.zeros((0, 4), np.float32)
    return {"frame": frame, "own": own, "swarm": swarm, "lab": lab, "score": score, "box": box}
