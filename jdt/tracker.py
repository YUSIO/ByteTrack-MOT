"""FCTrack fill-in with the joint track head in place of the separate verification network (Exp071).

The flow is that of fctrack.tracker.FCTrack with the Exp070 fallback placement ("formation first, the track's own
Kalman prediction when it has no neighbour"): after the unmodified ByteTrack update, every track lost for at most
`lookback` frames gets a window at its placement; the head, looking at the detector's stride-8 features of the
current frame, decides whether the track's UAV is in the window and regresses its box; accepted boxes re-activate the
track (same identity) and are written out. The head's template is the P3 feature at the track's last detector box,
taken in the frame of that box, so only causal information is used.
"""
import numpy as np
import torch

from fctrack.formation import estimate, select_neighbours
from fctrack.tracker import FCTrack, iou_matrix
from jdt.head import TG, decode_box, window_features
from jdt.query import make_query, stack
from torchvision.ops import roi_align


class JDTTrack(FCTrack):
    def __init__(self, tracker, STrack, cfg, head, device):
        super().__init__(tracker, STrack, cfg, verify=None)
        self.head, self.device = head, device
        self.tmpl = {}       # track id -> P3 roi (C x TG x TG) at its last detector box
        self.feat = None     # (P3, objectness logits, ratio) of the current frame

    def set_frame(self, p3, obj, ratio):
        self.feat = (p3, obj, ratio)

    def step(self, det, shape, image):
        rows = super().step(det, shape, image)
        f = self.frame
        upd = [(tid, m["box"]) for tid, m in self.meta.items() if m.get("real") == f]
        if upd:
            p3, _, r = self.feat
            tb = []
            for _, b in upd:
                c = (b[:2] + b[2:] / 2) * r
                s = max(1.5 * np.sqrt(b[2] * b[3]) * r, 16.0)
                tb.append([0.0, c[0] - s / 2, c[1] - s / 2, c[0] + s / 2, c[1] + s / 2])
            rois = torch.tensor(tb, dtype=p3.dtype, device=p3.device)
            tf = roi_align(p3, rois, TG, 1.0 / 8, sampling_ratio=2, aligned=True).float()
            for (tid, _), x in zip(upd, tf):
                self.tmpl[tid] = x
        return rows

    def recover(self, f, det, cur, W, H, image):
        r = self.rec
        p3, obj, ratio = self.feat
        cur_boxes = np.array(list(cur.values())).reshape(-1, 4)
        taken = cur_boxes[:, :2] + cur_boxes[:, 2:] / 2 if len(cur_boxes) else np.zeros((0, 2))
        sides = np.sqrt(cur_boxes[:, 2] * cur_boxes[:, 3]) if len(cur_boxes) else np.zeros(0)
        cands, queries = [], []
        for t in self.t.lost_stracks:
            m = self.meta.get(t.track_id)
            gap = f - t.frame_id
            if m is None or not t.is_activated or not 1 <= gap <= r["lookback"] or f - m["real"] > r["max_since_real"] or m["obs"] < r["min_observations"]:
                continue
            if t.track_id not in self.tmpl:
                continue
            box = m["box"]
            side = float(np.sqrt(box[2] * box[3]))
            kal = t.tlwh[:2] + t.tlwh[2:] / 2
            place, each, spread, used = self.placement(t, cur, box, side, kal)
            if not (0 <= place[0] < W and 0 <= place[1] < H):
                continue
            est_box = np.r_[place - box[2:] / 2, box[2:]]
            if len(cur_boxes) and iou_matrix(est_box, cur_boxes).max() >= r["iou_free"]:
                continue
            vel = np.asarray(t.mean[4:6], float)
            queries.append(make_query(centre=place, box=box, tbox=box, cv=kal, vel=vel, gap=gap, nb=each, spread=spread, occ=taken,
                                      member_sides=np.r_[sides, side], W=W, H=H, r=ratio))
            cands.append((t, used))
        self.stats["tried"] += len(cands)
        if not cands:
            return []
        q = stack(queries, self.device)
        idx = torch.zeros(len(cands), dtype=torch.long, device=self.device)
        fs, ob, _ = window_features(p3, obj, idx, q["centre"], q["half"])
        tf = torch.stack([self.tmpl[t.track_id] for t, _ in cands])
        with torch.no_grad():
            p, boxes = decode_box(self.head(fs, ob, tf, q), q)
        p, final = p.cpu().numpy().astype(float), boxes.cpu().numpy().astype(float) / ratio
        added, taken_boxes, done = [], cur_boxes, set()
        for i in np.argsort(-p):
            if p[i] < r["p_min"] or final[i, 2] <= 1 or final[i, 3] <= 1 or cands[i][0].track_id in done:
                continue
            if len(taken_boxes) and iou_matrix(final[i], taken_boxes).max() >= r["iou_free"]:
                continue
            t = cands[i][0]
            done.add(t.track_id)
            t.re_activate(self.STrack(final[i], r["score"]), f, new_id=False)
            self.t.lost_stracks = [x for x in self.t.lost_stracks if x is not t]
            self.t.tracked_stracks.append(t)
            self.snap[f][t.track_id] = final[i, :2] + final[i, 2:] / 2
            self.meta[t.track_id]["box"] = final[i].copy()
            taken_boxes = np.vstack((taken_boxes, final[i]))
            added.append([t.track_id, final[i].copy(), float(r["score"]), t])
            self.stats["recovered"] += 1
            self.stats["recovered_" + cands[i][1]] += 1
        return added

    def placement(self, t, cur, box, side, kal):
        """Formation estimate (Delaunay neighbours, offset model) or, without a neighbour, the Kalman prediction."""
        r = self.rec
        ref = self.snap.get(t.frame_id, {})
        if t.track_id in ref and r["estimator"] != "kalman":
            ids = list(ref)
            P0 = np.array([ref[i] for i in ids])
            avail = np.array([i in cur and i != t.track_id for i in ids])
            nb = select_neighbours(P0, ids.index(t.track_id), avail, r["graph"], r["max_neighbours"]) if avail.any() else []
            if len(nb) >= r["min_neighbours"]:
                N0 = P0[nb]
                N1 = np.array([cur[ids[j]][:2] + cur[ids[j]][2:] / 2 for j in nb])
                p0 = P0[ids.index(t.track_id)]
                c, spread, _ = estimate(p0, N0, N1, r["model"], r["consistency_sides"] * side)
                return c, N1 + (p0 - N0), spread, "formation"
            self.stats["no_neighbour"] += 1
        return kal, np.zeros((0, 2)), 0.0, "kalman"
