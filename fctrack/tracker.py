"""FCTrack on top of the unmodified ByteTrack association.

Every frame: (1) ByteTrack update with the detector's boxes; (2) for every track that has just gone missing, estimate
where the member is now, verify the estimate on the image and, if accepted, keep the box: the track is re-activated
with it and the box is written out; (3) optionally hold back formation outliers until the verifier has accepted them
on consecutive frames. With `outlier.order = "paper"` step (3) runs before step (2) on the associated tracks only, as in
the paper, and the members held back are not used as neighbours. Step (2) is where the graph (mst / knn / delaunay) and the motion model are switched; the
estimators `kalman` (the track's own prediction) and `hold` (its last box) are the controls without neighbours.
"""
from collections import Counter

import numpy as np

from fctrack.formation import OutlierFilter, PaperOutlierFilter, estimate, select_neighbours
from fctrack.relink import carry


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
    def __init__(self, tracker, STrack, cfg, verify=None, log=False):
        """cfg: the arm's `recover` and `outlier` blocks (either may be None); verify(img, tlwh) -> (p, boxes).

        log: keep one record per verified fill-in attempt in `self.attempts` (read only; no decision changes)."""
        self.t, self.STrack, self.verify = tracker, STrack, verify
        self.log, self.attempts = log, []
        self.rec, self.out = cfg.get("recover"), cfg.get("outlier")
        self.frame = 0
        self.snap = {}     # frame -> {track id: centre} of the members observed (or recovered) in that frame
        self.meta = {}     # track id -> last real frame, number of real observations, last box, outlier state
        if not self.out:
            self.filter = None
        elif self.out.get("form") == "paper":
            self.filter = PaperOutlierFilter(self.out["history"], self.out["percentile"], self.out.get("eps", 1.0))
        else:
            self.filter = OutlierFilter(self.out["history"], self.out["percentile"], self.out["regulariser"])
        self.first_filter = bool(self.out) and self.out.get("order") == "paper"
        self.stats = Counter()
        self.relab = cfg.get("relabel")  # Exp077: {"family", "tau", "gap"}; None keeps the tracker unchanged
        self.warps = {}                  # frame -> 2x3 camera motion (family gmc only; set by the runner)
        self.hist = {}                   # frame -> {track id: centre} of the rows written in that frame
        self.lastw = {}                  # track id -> (frame, centre, side) of its last written row
        self.head, self.features, self.seq = None, None, None  # Exp108: OnlineHead, Features and sequence name (set by the runner)
        self.bth = cfg.get("birth")      # Exp109: {"lo", "hi", "cover_iou", "rule": all | verifier | oracle, "p_min"}
        self.gt = {}                     # Exp109, rule oracle only (ceiling): frame -> GT tlwh rows (set by the runner)

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
        if self.first_filter and rows:  # the paper's order: outliers are removed before missing members are estimated
            rows = self.hold_back(f, rows, image)
            kept = {r[0] for r in rows}
            cur = {tid: b for tid, b in cur.items() if tid in kept}
        self.snap[f] = {tid: b[:2] + b[2:] / 2 for tid, b in cur.items()}
        if self.rec:
            rows += self.recover(f, det, cur, W, H, image)
            for old in [x for x in self.snap if x < f - self.rec["lookback"] - 1]:
                del self.snap[old]
        else:
            self.snap = {f: self.snap[f]}
        if self.head is not None:  # Exp108: claim uncovered weak boxes with the in-sequence head (before relabel, so a claim is a birth)
            rows += self.claim(f, det, rows, image)
        if self.bth is not None:  # Exp109: immediate birth of uncovered detections decided by a rule (before relabel)
            rows += self.birth(f, det, rows, image)
        if self.filter is not None and not self.first_filter and rows:
            rows = self.hold_back(f, rows, image)
        if self.relab:
            self.relabel_births(f, rows)
            self.hist[f] = {}
            for tid, box, _, _ in rows:
                if box[2] * box[3] > 0:
                    c = box[:2] + box[2:] / 2
                    self.hist[f][tid] = c
                    self.lastw[tid] = (f, c, float(np.sqrt(box[2] * box[3])))
            for old in [x for x in self.hist if x < f - self.relab["gap"] - 1]:
                del self.hist[old]
        if self.head is not None:  # pseudo-labels from the rows actually written in this frame
            written = [r for r in rows if r[1][2] * r[1][3] > 0]
            self.head.collect(f, det, written, lambda b: self.features.at((self.seq, f), image, b), iou_matrix)
        return [(tid, box, score) for tid, box, score, _ in rows]

    def birth(self, f, det, rows, image):
        """Exp109: an uncovered detection with score in [lo, hi) becomes a written, activated track in this frame when the
        rule says so (all: always; verifier: the FCTrack verifier's p >= p_min on the detection box; oracle: it matches at
        IoU >= 0.5 a GT member no written row matches, ceiling only). A ByteTrack track started on the same box in this
        frame (unconfirmed, not written) is activated instead of starting a second one. Highest score first."""
        b = self.bth
        if not len(det):
            return []
        written = [r for r in rows if r[1][2] * r[1][3] > 0]
        taken = np.array([r[1] for r in written]).reshape(-1, 4)
        tlwh = np.column_stack((det[:, :2], det[:, 2:4] - det[:, :2])).astype(float)
        cover = iou_matrix(tlwh, taken).max(1) if len(taken) else np.zeros(len(tlwh))
        sel = np.flatnonzero((det[:, 4] >= b["lo"]) & (det[:, 4] < b["hi"]) & (cover < b["cover_iou"]))
        if not len(sel):
            return []
        self.stats["birth_candidates"] += len(sel)
        if b["rule"] == "verifier":
            p, _ = self.verify(image(), tlwh[sel])
            sel = sel[np.asarray(p) >= b["p_min"]]
        elif b["rule"] == "oracle":
            g = self.gt.get(f, np.zeros((0, 4)))
            if len(g) and len(taken):
                from scipy.optimize import linear_sum_assignment
                m = iou_matrix(g, taken)
                gi, ti = linear_sum_assignment(-(m * (m >= 0.5)))
                g = np.delete(g, [x for x, y in zip(gi, ti) if m[x, y] >= 0.5], axis=0)
            if not len(g):
                return []
            from scipy.optimize import linear_sum_assignment
            m = iou_matrix(tlwh[sel], g)
            si, gi = linear_sum_assignment(-(m * (m >= 0.5)))
            sel = sel[[x for x, y in zip(si, gi) if m[x, y] >= 0.5]]
        added = []
        for k in sel[np.argsort(-det[sel, 4])]:
            box = tlwh[k]
            if len(taken) and iou_matrix(box, taken).max() >= b["cover_iou"]:
                continue
            same = [t for t in self.t.tracked_stracks if not t.is_activated and t.frame_id == f
                    and iou_matrix(box, t.tlwh).max() >= 0.7]
            if same:
                t = same[0]
                self.stats["birth_from_unconfirmed"] += 1
            else:
                t = self.STrack(box, det[k, 4])
                t.activate(self.t.kalman_filter, f)
                self.t.tracked_stracks.append(t)
            t.is_activated = True
            self.meta[t.track_id] = {"obs": 1, "outlier": False, "ok": 0, "cleared": False, "real": f, "box": box.copy()}
            self.snap.setdefault(f, {})[t.track_id] = box[:2] + box[2:] / 2
            taken = np.vstack((taken, box))
            added.append([t.track_id, box.copy(), det[k, 4], t])
            self.stats["born"] += 1
        return added

    def claim(self, f, det, rows, image):
        """Exp108: start an activated, written track on every claimed weak box (one box per place, best first)."""
        written = [r for r in rows if r[1][2] * r[1][3] > 0]
        idx, comb = self.head.claims(f, det, written, lambda b: self.features.at((self.seq, f), image, b), image, iou_matrix)
        taken = np.array([r[1] for r in written]).reshape(-1, 4)
        added = []
        for k in idx:
            box = np.r_[det[k, :2], det[k, 2:4] - det[k, :2]].astype(float)
            if len(taken) and iou_matrix(box, taken).max() >= self.head.c["cover_iou"]:
                continue
            t = self.STrack(box, det[k, 4])
            t.activate(self.t.kalman_filter, f)
            t.is_activated = True  # written now, like a member at the first frame; ByteTrack keeps it as a tracked track
            self.t.tracked_stracks.append(t)
            self.meta[t.track_id] = {"obs": 1, "outlier": False, "ok": 0, "cleared": False, "real": f, "box": box.copy()}
            self.snap.setdefault(f, {})[t.track_id] = box[:2] + box[2:] / 2
            taken = np.vstack((taken, box))
            added.append([t.track_id, box.copy(), det[k, 4], t])
            self.head.stats["claimed"] += 1
            self.stats["claimed"] += 1
        return added

    def relabel_births(self, f, rows):
        """Exp077, closed-loop form of Exp076, run on the rows of this frame after the fill-in (as the open-loop pass saw
        them): a track written for the first time takes the id of the nearest member that is not written now and was
        last written within `gap` frames, when that member's last centre carried to this frame (fctrack.relink.carry)
        is within `tau` sides of the new box. The old track is dropped from the tracker's Lost and Removed lists, so
        from now on the tracker and the fill-in see one track. Changes `rows` in place."""
        r = self.relab
        births = [k for k, row in enumerate(rows) if row[0] not in self.lastw and row[1][2] * row[1][3] > 0]
        if not births or f == 1:
            return
        now = {row[0] for row in rows}
        cur = {row[0]: row[1][:2] + row[1][2:] / 2 for row in rows if row[0] in self.lastw}
        pairs = []
        for i, k in enumerate(births):
            b = rows[k][1]
            c0, side = b[:2] + b[2:] / 2, float(np.sqrt(max(b[2] * b[3], 1.0)))
            for L, (f1, c1, side1) in self.lastw.items():
                if L in now or not 1 <= f - f1 <= r["gap"]:
                    continue
                p = carry(r["family"], c1, side1, f1, f, L, {f1: self.hist.get(f1, {}), f: cur}, self.warps)
                d = float(np.linalg.norm(p - c0)) / side
                if d <= r["tau"]:
                    pairs.append((d, i, L))
        done_i, done_L = set(), set()
        for d, i, L in sorted(pairs):
            if i in done_i or L in done_L:
                continue
            done_i.add(i); done_L.add(L)
            row = rows[births[i]]
            self.t.lost_stracks = [x for x in self.t.lost_stracks if x.track_id != L]
            self.t.removed_stracks = [x for x in self.t.removed_stracks if x.track_id != L]
            old = row[0]
            row[3].track_id = L
            row[0] = L
            if old in self.meta:  # the new track's own bookkeeping continues under the old id
                m = self.meta.pop(old)
                self.meta[L] = {**self.meta.get(L, {}), **{k: v for k, v in m.items() if k in ("real", "box")},
                                "obs": self.meta.get(L, {}).get("obs", 0) + m.get("obs", 0)}
            for snap in self.snap.values():
                if old in snap:
                    snap[L] = snap.pop(old)
            self.stats["relabelled"] += 1

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
        if self.log:  # state before any decision of this frame (re-activation below changes frame_id and covariance)
            pre = [(f - t.frame_id, f - self.meta[t.track_id]["real"], self.meta[t.track_id]["obs"],
                    float(t.covariance[0, 0]), float(t.covariance[1, 1])) for t, _, _ in cands]
        added, taken, done = [], cur_boxes, set()
        decision = ["below_p" if p[i] < r["p_min"] else "" for i in range(len(cands))]
        for i in np.argsort(-p):
            if p[i] < r["p_min"] or final[i, 2] <= 1 or final[i, 3] <= 1 or cands[i][0].track_id in done:
                decision[i] = decision[i] or ("done" if cands[i][0].track_id in done else "degenerate")
                continue
            if len(taken) and iou_matrix(final[i], taken).max() >= r["iou_free"]:
                decision[i] = "overlap"
                continue
            decision[i] = "written" if r.get("write", True) else "kept_unwritten"
            t = cands[i][0]
            done.add(t.track_id)
            t.re_activate(self.STrack(final[i], r["score"]), f, new_id=False)
            self.t.lost_stracks = [x for x in self.t.lost_stracks if x is not t]
            self.t.tracked_stracks.append(t)
            self.snap[f][t.track_id] = final[i, :2] + final[i, 2:] / 2
            self.meta[t.track_id]["box"] = final[i].copy() if r["use_regressed_box"] else self.meta[t.track_id]["box"]
            taken = np.vstack((taken, final[i]))
            if r.get("write", True):  # write = False: the fill-in only keeps the track alive inside the tracker (diagnostic)
                added.append([t.track_id, final[i].copy(), float(r["score"]), t])
            self.stats["recovered"] += 1
            self.stats["recovered_" + cands[i][2]] += 1
        if self.log:
            for i, (t, prop, used) in enumerate(cands):
                gap, since_real, obs, cxx, cyy = pre[i]
                self.attempts.append([f, t.track_id, gap, since_real, obs, used, *map(float, prop), *map(float, final[i]),
                                      float(p[i]), decision[i], cxx, cyy])
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
        if r["model"] == "offset_paper":  # squared pixels: a fraction of the box area (10 % in the paper)
            eps = r.get("consistency_area_fraction", 0.1) * float(size[0] * size[1])
        else:
            eps = r["consistency_sides"] * float(np.sqrt(size[0] * size[1]))
        c, _, used = estimate(P0[ids.index(t.track_id)], P0[nb], N1, r["model"], eps)
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
