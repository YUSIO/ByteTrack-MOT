"""Exp108: closed-loop form of the Exp106 in-sequence head (Idea 0037).

Per sequence, a logistic-regression head on frozen detector features is refit every `refit_every` frames on
pseudo-labels taken from this tracker's own written rows of earlier frames:
  positive = a written row whose track has been written on >= `positive_min_age` frames and whose box overlaps a
             detection of score >= `positive_det_score` at IoU >= `positive_det_iou` (at most `positive_per_frame`);
  negative = a detection of score < `negative_score_hi` with IoU < `negative_max_row_iou` to every written row
             (at most `negative_per_frame`).
The head scores the weak boxes (score in [lo, hi)) that no written row covers; a box is claimed when
logit(p) + logit(score) >= 0 (rule `head`), when score >= 0.5 (rule `score`, the control without a head), or with a
head fitted once elsewhere (rule `global`). Without a fitted head the `head` rule falls back to the score.
Features: L2(bilinear sample at the box centre) ++ L2(mean of the 3x3 cells around it) on one FPN level, as in Exp106.
"""
import numpy as np
import torch
import torch.nn.functional as F

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Features:
    """Frozen YOLOX backbone; one FPN level per frame, computed lazily."""

    def __init__(self, exp_file, ckpt, size, level, stride, device):
        from yolox.exp import get_exp
        exp = get_exp(str(exp_file), None)
        self.model = exp.get_model()
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
        self.model.to(device).eval()
        self.size, self.level, self.stride, self.device = tuple(size), level, stride, device
        self.key, self.feat, self.r = None, None, None

    def at(self, key, image, boxes):
        from yolox.data.data_augment import preproc
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        if not len(boxes):
            return np.zeros((0, 256), np.float32)
        if key != self.key:
            x, self.r = preproc(image(), self.size, MEAN, STD)
            with torch.no_grad():
                self.feat = self.model.backbone(torch.from_numpy(x).unsqueeze(0).to(self.device))[self.level].float()
            self.key = key
        feat, r, stride = self.feat, self.r, self.stride
        _, c, h, w = feat.shape
        cx = (boxes[:, 0] + boxes[:, 2] / 2) * r / stride - 0.5
        cy = (boxes[:, 1] + boxes[:, 3] / 2) * r / stride - 0.5
        g = torch.from_numpy(np.stack([2 * (cx + 0.5) / w - 1, 2 * (cy + 0.5) / h - 1], -1).astype(np.float32)).to(feat.device)
        with torch.no_grad():
            centre = F.grid_sample(feat, g.view(1, 1, -1, 2), mode="bilinear", align_corners=False)[0, :, 0].T
            pooled = F.avg_pool2d(F.pad(feat, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
            ci = np.clip(np.floor(cy + 0.5).astype(int), 0, h - 1)
            cj = np.clip(np.floor(cx + 0.5).astype(int), 0, w - 1)
            win = pooled[0][:, torch.from_numpy(ci), torch.from_numpy(cj)].T
            out = torch.cat([F.normalize(centre, dim=1), F.normalize(win, dim=1)], 1)
        # float16 round trip, as the Exp106 feature files were stored
        return out.cpu().numpy().astype(np.float16).astype(np.float32)


class OnlineHead:
    def __init__(self, cfg, seed, global_model=None):
        self.c, self.rng, self.glob = cfg, np.random.default_rng(seed), global_model
        self.X, self.y, self.f = [], [], []   # samples and the frame they come from
        self.model, self.fit_at = None, None
        self.age = {}                         # track id -> number of written frames
        self.stats = {"fits": 0, "claimed": 0, "scored": 0, "scored_without_head": 0}

    def _fit(self, frame):
        from sklearn.linear_model import LogisticRegression
        c = self.c
        if not self.X:
            self.model = None
            return
        f = np.concatenate(self.f)
        keep = f < frame
        y = np.concatenate(self.y)[keep]
        if y.sum() < c["min_pos"] or (1 - y).sum() < c["min_neg"]:
            self.model = None
            return
        X = np.concatenate(self.X)[keep]
        self.model = LogisticRegression(C=c["C"], class_weight=c["class_weight"], max_iter=c["max_iter"]).fit(X, y)
        self.stats["fits"] += 1

    def claims(self, frame, det, rows, feats, image, iou):
        """det: n x 5 (x1, y1, x2, y2, score); rows: written rows of this frame [(id, tlwh, ...)].
        Returns the indices of `det` to claim, best first, and their combined scores."""
        c = self.c
        if (frame - 1) % c["refit_every"] == 0 and self.fit_at != frame and c["rule"] == "head":
            self.fit_at = frame
            self._fit(frame)
        tlwh = np.column_stack((det[:, :2], det[:, 2:4] - det[:, :2])) if len(det) else np.zeros((0, 4))
        rb = np.array([r[1] for r in rows]).reshape(-1, 4)
        cover = iou(tlwh, rb).max(1) if len(rb) and len(tlwh) else np.zeros(len(tlwh))
        sel = np.flatnonzero((det[:, 4] >= c["lo"]) & (det[:, 4] < c["hi"]) & (cover < c["cover_iou"])) if len(det) else np.zeros(0, int)
        if not len(sel):
            return [], []
        s = logit(det[sel, 4])
        if c["rule"] == "score":
            comb = s
        else:
            m = self.model if c["rule"] == "head" else self.glob
            self.stats["scored"] += len(sel)
            if m is None:
                self.stats["scored_without_head"] += len(sel)
                comb = s
            else:
                comb = logit(m.predict_proba(feats(tlwh[sel]))[:, 1]) + s
        order = np.argsort(-comb)
        return [int(sel[i]) for i in order if comb[i] >= 0], [float(comb[i]) for i in order if comb[i] >= 0]

    def collect(self, frame, det, rows, feats, iou):
        """Pseudo-labels from this frame's written rows (only the `head` rule learns online)."""
        c = self.c
        for r in rows:
            self.age[r[0]] = self.age.get(r[0], 0) + 1
        if c["rule"] != "head" or not len(det):
            return
        tlwh = np.column_stack((det[:, :2], det[:, 2:4] - det[:, :2]))
        rb = np.array([r[1] for r in rows]).reshape(-1, 4)
        pos = np.zeros((0, 4))
        if len(rb):
            strong = tlwh[det[:, 4] >= c["positive_det_score"]]
            ok = iou(rb, strong).max(1) >= c["positive_det_iou"] if len(strong) else np.zeros(len(rb), bool)
            ok &= np.array([self.age.get(r[0], 0) >= c["positive_min_age"] for r in rows])
            idx = self.rng.permutation(np.flatnonzero(ok))[:c["positive_per_frame"]]
            pos = rb[idx]
        far = iou(tlwh, rb).max(1) < c["negative_max_row_iou"] if len(rb) else np.ones(len(tlwh), bool)
        nidx = np.flatnonzero((det[:, 4] < c["negative_score_hi"]) & far)
        neg = tlwh[self.rng.permutation(nidx)[:c["negative_per_frame"]]]
        if not len(pos) and not len(neg):
            return
        X = feats(np.vstack((pos, neg)))
        self.X.append(X)
        self.y.append(np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)])
        self.f.append(np.full(len(X), frame))
