"""FCTrack on top of the unmodified ByteTrack association.

Every frame: (1) ByteTrack update with the detector's boxes; (2) for every track that has just gone missing, estimate
where the member is now, verify the estimate on the image and, if accepted, keep the box: the track is re-activated
with it and the box is written out; (3) optionally hold back formation outliers until the verifier has accepted them
on consecutive frames. Step (2) is where the graph (mst / knn / delaunay) and the motion model are switched; the
estimators `kalman` (the track's own prediction) and `hold` (its last box) are the controls without neighbours.
"""
from collections import Counter

import numpy as np

from fctrack.formation import OutlierFilter, estimate, select_neighbours


def wrap_stracks(STrack):
    """Remember the detector box a track was last updated with (ByteTrack itself only keeps the filtered state)."""
    if getattr(STrack, "_fctrack_wrapped", False):
        return
    upd, rea, act = STrack.update, STrack.re_activate, STrack.activate

    def update_w(self, new, fid):
        self._det = new._tlwh.copy()
        return upd(self, new, fid)

    def re_activate_w(self, new, fid, new_id=False):
        self._det = new._tlwh.copy()
        return rea(self, new, fid, new_id)

    def activate_w(self, kf, fid):
        self._det = self._tlwh.copy()
        return act(self, kf, fid)

    STrack.update, STrack.re_activate, STrack.activate = update_w, re_activate_w, activate_w
    STrack._fctrack_wrapped = True


def iou_matrix(a, b):
    """IoU of tlwh boxes."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, :2] + a[:, None, 2:], b[None, :, :2] + b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    return inter / np.maximum(np.prod(a[:, 2:], axis=1)[:, None] + np.prod(b[:, 2:], axis=1)[None, :] - inter, 1e-9)


class FCTrack:
    def __init__(self, tracker, STrack, cfg, verify=None):
        """cfg: the arm's `recover` and `outlier` blocks (either may be None); verify(img, tlwh) -> (p, boxes)."""
        self.t, self.STrack, self.verify = tracker, STrack, verify
        self.rec, self.out = cfg.get("recover"), cfg.get("outlier")
        self.frame = 0
        self.snap = {}     # frame -> {track id: centre} of the members observed (or recovered) in that frame
        self.meta = {}     # track id -> last real frame, number of real observations, last box, outlier state
        self.filter = OutlierFilter(self.out["history"], self.out["percentile"], self.out["regulariser"]) if self.out else None
        self.stats = Counter()

    def step(self, det, shape, image):
        """det: n x 5 (x1, y1, x2, y2, score); image: callable returning the frame. Returns rows (id, tlwh, score)."""
        self.frame += 1
        f, (H, W) = self.frame, shape
        outs = self.t.update(det.copy(), shape, shape)
        rows, cur = [], {}
        for t in outs:
            box = np.asarray(t._det if t.frame_id == f else t.tlwh, float)
            rows.append([t.track_id, box, t.score, t])  # the score keeps its dtype so that the written text matches the stored baseline
            if t.frame_id == f:
                cur[t.track_id] = box
                m = self.meta.setdefault(t.track_id, {"obs": 0, "outlier": False, "ok": 0, "cleared": False})
                m.update(real=f, obs=m["obs"] + 1, box=box)
        self.snap[f] = {tid: b[:2] + b[2:] / 2 for tid, b in cur.items()}
        if self.rec:
            rows += self.recover(f, det, cur, W, H, image)
            for old in [x for x in self.snap if x < f - self.rec["lookback"] - 1]:
                del self.snap[old]
        else:
            self.snap = {f: self.snap[f]}
        if self.filter is not None and rows:
            rows = self.hold_back(f, rows, image)
        return [(tid, box, score) for tid, box, score, _ in rows]

    def recover(self, f, det, cur, W, H, image):
        r = self.rec
        cur_boxes = np.array(list(cur.values())).reshape(-1, 4)
        usable = det[det[:, 4] > 0.1]
        usable_tlwh = np.column_stack((usable[:, :2], usable[:, 2:4] - usable[:, :2])) if len(usable) else np.zeros((0, 4))
        cands = []
        for t in self.t.lost_stracks:
            m = self.meta.get(t.track_id)
            gap = f - t.frame_id
            if m is None or not t.is_activated or not 1 <= gap <= r["lookback"] or f - m["real"] > r["max_since_real"] or m["obs"] < r["min_observations"]:
                continue
            size = m["box"][2:]
            for spec in r["estimator"].split("+"):  # "a+b": both positions are tried, the better verified one is kept
                for est in spec.split("|"):         # "a|b": b only when a has no answer (no neighbour)
                    got = self.position(t, m, est, cur, size)
                    if got is not None:
                        break
                if got is None or not (0 <= got[0][0] < W and 0 <= got[0][1] < H):
                    continue
                cands.append((t, np.r_[got[0] - size / 2, size], got[1]))
        if not cands:
            return []
        boxes = np.array([b for _, b, _ in cands])
        keep = np.ones(len(cands), bool)
        if len(cur_boxes):  # never on top of a member that is being tracked
            keep &= iou_matrix(boxes, cur_boxes).max(1) < r["iou_free"]
        if r["skip_if_detected"] and len(usable_tlwh):  # the detector already has a box there: leave it to the association
            keep &= iou_matrix(boxes, usable_tlwh).max(1) < r["iou_free"]
        cands, boxes = [c for c, k in zip(cands, keep) if k], boxes[keep]
        self.stats["tried"] += len(cands)
        if not cands:
            return []
        if self.verify is not None:
            p, final = self.verify(image(), boxes)
            for _ in range(r.get("iterations", 1) - 1):  # look again at the regressed box
                p, final = self.verify(image(), final)
            if not r["use_regressed_box"]:
                final = boxes
        else:
            p, final = np.ones(len(cands)), boxes
        added, taken, done = [], cur_boxes, set()
        for i in np.argsort(-p):
            if p[i] < r["p_min"] or final[i, 2] <= 1 or final[i, 3] <= 1 or cands[i][0].track_id in done:
                continue
            if len(taken) and iou_matrix(final[i], taken).max() >= r["iou_free"]:
                continue
            t = cands[i][0]
            done.add(t.track_id)
            t.re_activate(self.STrack(final[i], r["score"]), f, new_id=False)
            self.t.lost_stracks = [x for x in self.t.lost_stracks if x is not t]
            self.t.tracked_stracks.append(t)
            self.snap[f][t.track_id] = final[i, :2] + final[i, 2:] / 2
            self.meta[t.track_id]["box"] = final[i].copy() if r["use_regressed_box"] else self.meta[t.track_id]["box"]
            taken = np.vstack((taken, final[i]))
            added.append([t.track_id, final[i].copy(), float(r["score"]), t])
            self.stats["recovered"] += 1
            self.stats["recovered_" + cands[i][2]] += 1
        return added

    def position(self, t, m, est, cur, size):
        """Centre of the missing member under one estimator, with the name of what was used; None when it has no answer."""
        r = self.rec
        if est == "kalman":
            return t.tlwh[:2] + t.tlwh[2:] / 2, "kalman"
        if est == "hold":
            return m["box"][:2] + size / 2, "hold"
        ref = self.snap.get(t.frame_id, {})
        if t.track_id not in ref:
            return None
        ids = list(ref)
        P0 = np.array([ref[i] for i in ids])
        avail = np.array([i in cur for i in ids])
        nb = select_neighbours(P0, ids.index(t.track_id), avail, r["graph"], r["max_neighbours"])
        if len(nb) < r["min_neighbours"]:
            self.stats["no_neighbour"] += 1
            return None
        N1 = np.array([cur[ids[j]][:2] + cur[ids[j]][2:] / 2 for j in nb])
        c, _, used = estimate(P0[ids.index(t.track_id)], P0[nb], N1, r["model"], r["consistency_sides"] * float(np.sqrt(size[0] * size[1])))
        return c, used

    def hold_back(self, f, rows, image):
        o = self.out
        centres = np.array([b[:2] + b[2:] / 2 for _, b, _, _ in rows])
        flag = self.filter.update(centres)
        kept, check = [], []
        for row, bad in zip(rows, flag):
            m = self.meta.setdefault(row[0], {"obs": 0, "outlier": False, "ok": 0, "cleared": False, "real": f, "box": row[1]})
            if bad and not m["cleared"] and not m["outlier"]:
                m["outlier"], m["ok"] = True, 0
                self.stats["flagged"] += 1
            if m["outlier"]:
                check.append(row)
            else:
                kept.append(row)
        if check:
            p = self.verify(image(), np.array([r[1] for r in check]))[0] if self.verify is not None else np.zeros(len(check))
            for row, q in zip(check, p):
                m = self.meta[row[0]]
                m["ok"] = m["ok"] + 1 if q >= o["p_min"] else 0
                if m["ok"] >= o["restore_frames"]:  # appearance consistent on consecutive frames: a member after all
                    m["outlier"], m["cleared"] = False, True
                    kept.append(row)
                    self.stats["restored"] += 1
                else:
                    self.stats["held_back_rows"] += 1
        return kept
