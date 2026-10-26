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
        self.trace = None  # Exp148: optional read-only observer; never used for decisions
        self.point_provider = None  # Exp147: optional causal image correspondence; default keeps A unchanged
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
        self.birth_log = []              # Exp115: one record per born candidate (frame, score, verifier p, written box, track)
        self.verify_birth = None         # Exp123 diagnostic: a separate verifier for the birth decision (None = self.verify)
        self.log_birth_low = None        # Exp124: callable(img, tlwh) that only records uncovered candidates in [log_lo, lo) (no decision)
        self.weak_birth = cfg.get("weak_birth")  # Separate supplemental channel; never rescore BYTE input.
        self.weak_birth_log = []
        if self.weak_birth is not None:
            if self.bth is None or not (0 <= self.weak_birth["lo"] < self.weak_birth["hi"] <= self.bth["lo"]):
                raise ValueError("weak_birth must be disjoint from the existing birth score band")

    def step(self, det, shape, image):
        """det: n x 5 (x1, y1, x2, y2, score); image: callable returning the frame. Returns rows (id, tlwh, score)."""
        self.frame += 1
        f, (H, W) = self.frame, shape
        trace = self.trace
        if trace is not None:
            trace.begin(f)
            trace.record("pre_byte", pools=trace.pools(self.t, self.meta), detections_xyxy_score=det)
        outs = self.t.update(det.copy(), shape, shape)
        if trace is not None:
            trace.record("post_byte", pools=trace.pools(self.t, self.meta), output_uids=[trace.uid(t) for t in outs])
        rows, cur = [], {}
        for t in outs:
            box = np.asarray(t._det if t.frame_id == f else t.tlwh, float)
            rows.append([t.track_id, box, t.score, t])  # the score keeps its dtype so that the written text matches the stored baseline
            if t.frame_id == f:
                cur[t.track_id] = box
                if self.point_provider is not None:
                    # Only BYTETracker's detector-updated outputs are anchors. Recovery
                    # can overwrite meta.box; never pair that box with an older real frame.
                    t._visual_anchor_frame = f
                    t._visual_anchor_box = box.copy()
                m = self.meta.setdefault(t.track_id, {"obs": 0, "outlier": False, "ok": 0, "cleared": False})
                m.update(real=f, obs=m["obs"] + 1, box=box)
                if m.get("prov", 0) > 0 and m["real"] == f and m["obs"] > 1:  # Exp111: a real detection after the birth frame
                    m["prov"] -= 1
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
        if self.weak_birth is not None:
            rows += self.supplement_weak_birth(f, det, rows, image)
        if self.filter is not None and not self.first_filter and rows:
            rows = self.hold_back(f, rows, image)
        if self.relab:
            if trace is not None:
                before_relabel = [dict(uid=trace.uid(t), row_id=tid, object_id=t.track_id) for tid, _, _, t in rows]
            self.relabel_births(f, rows)
            if trace is not None:
                trace.record("relabel", before=before_relabel,
                             after=[dict(uid=trace.uid(t), row_id=tid, object_id=t.track_id) for tid, _, _, t in rows])
            self.hist[f] = {}
            for tid, box, _, _ in rows:
                if box[2] * box[3] > 0:
                    c = box[:2] + box[2:] / 2
                    self.hist[f][tid] = c
                    self.lastw[tid] = (f, c, float(np.sqrt(box[2] * box[3])))
            for old in [x for x in self.hist if x < f - self.relab["gap"] - 1]:
                del self.hist[old]
        for e in self.birth_log:  # Exp115: the id a born track carries after this step's relabel
            if "id" not in e:
                e["id"] = e["t"].track_id
        for e in self.weak_birth_log:
            if "public_id" not in e:
                e["public_id"] = e.pop("track").track_id
        if self.head is not None:  # pseudo-labels from the rows actually written in this frame
            written = [r for r in rows if r[1][2] * r[1][3] > 0]
            self.head.collect(f, det, written, lambda b: self.features.at((self.seq, f), image, b), iou_matrix)
        if trace is not None:
            trace.record("final", pools=trace.pools(self.t, self.meta),
                         rows=[dict(uid=trace.uid(t), track_id=tid, tlwh=box, score=score) for tid, box, score, t in rows])
            trace.finish()
        return [(tid, box, score) for tid, box, score, _ in rows]

    def supplement_weak_birth(self, f, det, rows, image):
        """Reuse birth activation after the original channel, before relabel.

        Oracle uses only current GT for acceptance, never GT coordinates/IDs as outputs.
        Ordinary association, recovery and strong births always run first.
        """
        original = self.bth
        before = self.stats.copy()
        self.bth = self.weak_birth
        try:
            added = self.birth(f, det, rows, image)
        finally:
            self.bth = original
        for key in ("birth_candidates", "born", "birth_from_unconfirmed"):
            self.stats["weak_" + key] += self.stats[key] - before[key]
        for tid, box, score, track in added:
            self.weak_birth_log.append({"frame": f, "initial_id": tid, "box": box.copy(),
                                        "score": float(score), "track": track})
        return added

    def birth(self, f, det, rows, image):
        """Exp109: an uncovered detection with score in [lo, hi) becomes a written, activated track in this frame when the
        rule says so (all: always; verifier: the FCTrack verifier's p >= p_min on the detection box; oracle: it matches at
        IoU >= 0.5 a GT member no written row matches, ceiling only). A ByteTrack track started on the same box in this
        frame (unconfirmed, not written) is activated instead of starting a second one. Highest score first."""
        b = self.bth
        trace = self.trace
        if not len(det):
            return []
        written = [r for r in rows if r[1][2] * r[1][3] > 0]
        taken = np.array([r[1] for r in written]).reshape(-1, 4)
        tlwh = np.column_stack((det[:, :2], det[:, 2:4] - det[:, :2])).astype(float)
        cover = iou_matrix(tlwh, taken).max(1) if len(taken) else np.zeros(len(tlwh))
        sel = np.flatnonzero((det[:, 4] >= b["lo"]) & (det[:, 4] < b["hi"]) & (cover < b["cover_iou"]))
        if trace is not None:
            trace.record("birth_selection", selected_detection_indices=sel, cover=cover,
                         lo=b["lo"], hi=b["hi"], cover_iou=b["cover_iou"], rule=b["rule"])
        if self.log_birth_low is not None and b.get("log_lo") is not None:  # Exp124: record the lower band, decide nothing
            low = np.flatnonzero((det[:, 4] >= b["log_lo"]) & (det[:, 4] < b["lo"]) & (cover < b["cover_iou"]))
            if len(low):
                self.log_birth_low(image(), tlwh[low])
        if not len(sel):
            return []
        self.stats["birth_candidates"] += len(sel)
        final = tlwh.copy()
        if b["rule"] == "verifier":
            p, reg = (self.verify_birth or self.verify)(image(), tlwh[sel])
            if b.get("use_regressed_box"):  # Exp111: the same use of the verifier as the fill-in (write its regressed box)
                final[sel] = np.asarray(reg, float)
            ok = (np.asarray(p) >= b["p_min"]) & (final[sel, 2] > 1) & (final[sel, 3] > 1)
            if trace is not None:
                trace.record("birth_verification", detection_indices=sel, proposals=tlwh[sel], p=p,
                             regressed=reg, final=final[sel], accepted=ok, p_min=b["p_min"])
            sel = sel[ok]
        elif b["rule"] in ("policy", "all_logged"):
            # Exp115: every uncovered candidate >= lo is scored by the verifier; the written box is the regressed one (as
            # use_regressed_box). policy: born when a * logit(p) + b * logit(score) + tau >= 0; all_logged: all are born (value collection).
            pv, reg = (self.verify_birth or self.verify)(image(), tlwh[sel])
            pv = np.asarray(pv, float)
            final[sel] = np.asarray(reg, float)
            ok = (final[sel, 2] > 1) & (final[sel, 3] > 1)
            if b["rule"] == "policy":
                lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
                ok &= (b["a"] * lg(pv) + b["b"] * lg(det[sel, 4].astype(float)) + b["tau"]) >= 0
            pmap = {int(k): float(q) for k, q in zip(sel, pv)}
            sel = sel[ok]
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
            box = final[k]
            if len(taken) and iou_matrix(box, taken).max() >= b["cover_iou"]:
                if trace is not None:
                    trace.record("birth_decision", detection_index=int(k), final=box, decision="final_overlap")
                continue
            same = [t for t in self.t.tracked_stracks if not t.is_activated and t.frame_id == f
                    and iou_matrix(tlwh[k], t.tlwh).max() >= 0.7]
            if same:
                t = same[0]
                if b.get("use_regressed_box"):  # its state starts from the written (regressed) box, as a fill-in re-activation does
                    t.re_activate(self.STrack(box, det[k, 4]), f, new_id=False)
                self.stats["birth_from_unconfirmed"] += 1
            else:
                t = self.STrack(box, det[k, 4])
                t.activate(self.t.kalman_filter, f)
                self.t.tracked_stracks.append(t)
            t.is_activated = True
            self.meta[t.track_id] = {"obs": 1, "outlier": False, "ok": 0, "cleared": False, "real": f, "box": box.copy(),
                                     "prov": int(b.get("probation_obs", 0))}  # Exp111: real detections still needed before fill-in may keep it alive
            self.snap.setdefault(f, {})[t.track_id] = box[:2] + box[2:] / 2
            taken = np.vstack((taken, box))
            added.append([t.track_id, box.copy(), det[k, 4], t])
            if trace is not None:
                trace.record("birth_decision", detection_index=int(k), final=box, decision="written",
                             reused_unconfirmed=bool(same), track=trace.track(t, self.meta))
            if b["rule"] in ("policy", "all_logged"):
                self.birth_log.append({"frame": f, "score": float(det[k, 4]), "p": pmap[int(k)], "box": box.copy(), "t": t})
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
                self.meta[L] = {**self.meta.get(L, {}), **{k: v for k, v in m.items() if k in ("real", "box", "prov")},
                                "obs": self.meta.get(L, {}).get("obs", 0) + m.get("obs", 0)}
            for snap in self.snap.values():
                if old in snap:
                    snap[L] = snap.pop(old)
            self.stats["relabelled"] += 1

    def recover(self, f, det, cur, W, H, image):
        r = self.rec
        trace = self.trace
        cur_boxes = np.array(list(cur.values())).reshape(-1, 4)
        usable = det[det[:, 4] > 0.1]
        usable_tlwh = np.column_stack((usable[:, :2], usable[:, 2:4] - usable[:, :2])) if len(usable) else np.zeros((0, 4))
        cands = []
        for t in self.t.lost_stracks:
            m = self.meta.get(t.track_id)
            gap = f - t.frame_id
            if trace is not None:
                reasons = []
                if m is None:
                    reasons.append("missing_meta")
                if not t.is_activated:
                    reasons.append("not_activated")
                if not 1 <= gap <= r["lookback"]:
                    reasons.append("gap_outside_lookback")
                if m is not None:
                    if f - m["real"] > r["max_since_real"]:
                        reasons.append("since_real_exceeded")
                    if m["obs"] < r["min_observations"]:
                        reasons.append("too_few_observations")
                    if m.get("prov", 0) > 0:
                        reasons.append("birth_probation")
                trace.record("recovery_eligibility", track=trace.track(t, self.meta), gap=gap,
                             since_real=None if m is None else f - m["real"], reasons=reasons,
                             eligible=not reasons, lookback=r["lookback"], max_since_real=r["max_since_real"],
                             min_observations=r["min_observations"])
            if m is None or not t.is_activated or not 1 <= gap <= r["lookback"] or f - m["real"] > r["max_since_real"] or m["obs"] < r["min_observations"]:
                continue
            if m.get("prov", 0) > 0:  # Exp111: provisional (verifier-born, not yet confirmed again by the detector)
                self.stats["fill_in_skipped_probation"] += 1
                continue
            size = m["box"][2:]
            for spec in r["estimator"].split("+"):  # "a+b": both positions are tried, the better verified one is kept
                for est in spec.split("|"):         # "a|b": b only when a has no answer (no neighbour)
                    got = self.position(t, m, est, cur, size)
                    if trace is not None:
                        trace.record("recovery_position", uid=trace.uid(t), track_id=t.track_id, slot=spec,
                                     requested_estimator=est, result=None if got is None else dict(centre=got[0], used=got[1]))
                    if got is not None:
                        break
                if got is None or not (0 <= got[0][0] < W and 0 <= got[0][1] < H):
                    if trace is not None:
                        trace.record("recovery_proposal", uid=trace.uid(t), track_id=t.track_id, slot=spec,
                                     decision="no_position" if got is None else "outside_image")
                    continue
                cands.append((t, np.r_[got[0] - size / 2, size], got[1]))
                if trace is not None:
                    trace.record("recovery_proposal", uid=trace.uid(t), track_id=t.track_id, slot=spec,
                                 proposal=cands[-1][1], used=got[1], decision="proposed")
        if not cands:
            return []
        self.stats["proposals_before_overlap"] += len(cands)
        unique = {(id(t), *map(float, b)) for t, b, _ in cands}
        self.stats["unique_proposals_before_overlap"] += len(unique)
        boxes = np.array([b for _, b, _ in cands])
        keep = np.ones(len(cands), bool)
        if len(cur_boxes):  # never on top of a member that is being tracked
            keep &= iou_matrix(boxes, cur_boxes).max(1) < r["iou_free"]
        if r["skip_if_detected"] and len(usable_tlwh):  # the detector already has a box there: leave it to the association
            keep &= iou_matrix(boxes, usable_tlwh).max(1) < r["iou_free"]
        if trace is not None:
            trace.record("recovery_prefilter", candidates=[dict(track=trace.track(t, self.meta), proposal=b, estimator=e)
                                                          for t, b, e in cands],
                         keep=keep, current_boxes=cur_boxes, usable_detection_boxes=usable_tlwh,
                         iou_free=r["iou_free"], skip_if_detected=r["skip_if_detected"])
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
        if trace is not None:
            trace.record("recovery_verification", candidates=[dict(track=trace.track(t, self.meta), proposal=b, estimator=e)
                                                             for t, b, e in cands],
                         p=p, final=final, p_min=r["p_min"])
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
        if trace is not None:
            trace.record("recovery_decision", candidates=[dict(uid=trace.uid(t), track_id=t.track_id, proposal=prop,
                                                              final=final[i], p=p[i], estimator=used, decision=decision[i])
                                                         for i, (t, prop, used) in enumerate(cands)])
        if self.log:
            for i, (t, prop, used) in enumerate(cands):
                gap, since_real, obs, cxx, cyy = pre[i]
                self.attempts.append([f, t.track_id, gap, since_real, obs, used, *map(float, prop), *map(float, final[i]),
                                      float(p[i]), decision[i], cxx, cyy])
        return added

    def position(self, t, m, est, cur, size):
        """Centre of the missing member under one estimator, with the name of what was used; None when it has no answer."""
        r = self.rec
        if est == "cotracker":
            if self.point_provider is None:
                raise ValueError("cotracker estimator requires a point provider")
            centre = self.point_provider(self.frame, t)
            return None if centre is None else (centre, "cotracker")
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
