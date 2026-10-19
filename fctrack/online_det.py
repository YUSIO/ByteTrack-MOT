"""Exp116: a YOLOX detector in the tracking loop whose objectness layers are adapted online from the tracker's own output.

Per sequence the three 1 x 1 objectness layers of the YOLOX head (`head.obj_preds[k]`, 128 -> 1, one per FPN level) start from
the source weights and are updated by a few Adam steps per frame on a replay buffer of (cell feature, label) pairs built from
the rows the tracker wrote in the frames already processed (causal: frame t's rows only change the detections of frames > t):
  positive  the cell containing the centre of a written row (rows of tracks written for >= `pos_age` frames), weight 1, or
            `w_blind` when no detection of score >= `blind_score` supports the row (the row exists because the verifier and the
            formation estimate kept it: the signal the detector does not have);
  negative  the `n_hard` highest-scoring cells (score >= `hard_score`) away from every row, weight `w_hard`; and `n_rand` random
            cells away from every row, weight `w_rand`.
An L2-SP penalty `lam * ||theta - theta_0||^2` ties the layers to the source weights. Everything else (backbone, neck, regression
and classification branches) is frozen, so the detector's boxes come from the source model; only the scores move.
With `lr = 0` (or adapt = False) the detections equal the cached ones of the exporter (same preproc, conf 0.001, NMS 0.7,
text rounding of the cache).
"""
import numpy as np
import torch
import torch.nn.functional as F

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class OnlineDetector:
    def __init__(self, exp_file, ckpt, device, size=(1088, 1088), conf=0.001, nms=0.7, adapt=None):
        from yolox.exp import get_exp
        exp = get_exp(str(exp_file), None)
        self.model = exp.get_model()
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.exp, self.device, self.size, self.conf, self.nms = exp, device, tuple(size), conf, nms
        self.head = self.model.head
        self.strides = list(self.head.strides)
        self.W0 = [self.head.obj_preds[k].weight.detach().clone().view(1, -1) for k in range(3)]
        self.b0 = [self.head.obj_preds[k].bias.detach().clone().view(1) for k in range(3)]
        self.cfg = adapt or {"lr": 0.0}
        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self):
        self.W = [w.clone().requires_grad_(True) for w in self.W0]
        self.b = [b.clone().requires_grad_(True) for b in self.b0]
        c = self.cfg
        self.opt = torch.optim.Adam(self.W + self.b, lr=c["lr"]) if c.get("lr", 0) > 0 else None
        self.buf = [{"x": [], "y": [], "w": []} for _ in range(3)]
        self.frame = 0
        self.rng = np.random.default_rng(c.get("seed", 0))
        self.pos_cells = [0, 0, 0]  # positive cells seen per level, for the prior correction of the balanced batches
        self.stats = {"updates": 0, "pos": 0, "blind_pos": 0, "hard_neg": 0, "rand_neg": 0}
        self.cache = None

    # ------------------------------------------------------------ detection
    @torch.no_grad()
    def detect(self, img):
        from yolox.data.data_augment import preproc
        from yolox.utils import postprocess
        x, r = preproc(img, self.size, MEAN, STD)
        feats = self.model.backbone(torch.from_numpy(x).unsqueeze(0).to(self.device))
        h = self.head
        outs, regf, shapes, logits = [], [], [], []
        for k, (cls_conv, reg_conv, xin) in enumerate(zip(h.cls_convs, h.reg_convs, feats)):
            z = h.stems[k](xin)
            cls_feat, reg_feat = cls_conv(z), reg_conv(z)
            cls_out, reg_out = h.cls_preds[k](cls_feat), h.reg_preds[k](reg_feat)
            obj_logit = F.conv2d(reg_feat, self.W[k].detach().view(1, -1, 1, 1), self.b[k].detach())
            outs.append(torch.cat([reg_out, obj_logit.sigmoid(), cls_out.sigmoid()], 1))
            regf.append(reg_feat)
            logits.append((obj_logit, cls_out))
            shapes.append(reg_feat.shape[-2:])
        h.hw = [o.shape[-2:] for o in outs]
        flat = torch.cat([o.flatten(start_dim=2) for o in outs], dim=2).permute(0, 2, 1)
        dec = h.decode_outputs(flat, dtype=feats[0].type())
        out = postprocess(dec, self.exp.num_classes, self.conf, self.nms)[0]
        # per-cell score (objectness x classification) for hard negatives, level by level
        cell_score = [(ol.sigmoid() * co.sigmoid())[0, 0] for ol, co in logits]
        self.cache = {"regf": regf, "cell_score": cell_score, "r": r, "shapes": shapes}
        if out is None:
            return np.zeros((0, 5), np.float32)
        o = out.cpu().numpy()
        b = o[:, :4] / r
        sc = o[:, 4] * o[:, 5]
        # same text rounding as the exporter's det.txt (x y w h with 4 decimals, score with 6), then the runner's conversion
        x1, y1 = np.round(b[:, 0], 4), np.round(b[:, 1], 4)
        w, hh = np.round(b[:, 2] - b[:, 0], 4), np.round(b[:, 3] - b[:, 1], 4)
        return np.column_stack((x1, y1, x1 + w, y1 + hh, np.round(sc, 6))).astype(np.float32)

    # ----------------------------------------------------------- adaptation
    @torch.no_grad()
    def _collect(self, rows, det):
        """Build the (level, feature, label, weight) samples of the frame just tracked."""
        c, ch = self.cfg, self.cache
        r, regf = ch["r"], ch["regf"]
        centres = []
        for box, age in rows:
            if age < c.get("pos_age", 2):
                continue
            x, y, w, h = box
            if w * h <= 0:
                continue
            cx, cy, side = (x + w / 2) * r, (y + h / 2) * r, float(np.sqrt(w * h)) * r
            blind = True
            if len(det):
                ix1, iy1 = np.maximum(det[:, 0], x), np.maximum(det[:, 1], y)
                ix2, iy2 = np.minimum(det[:, 2], x + w), np.minimum(det[:, 3], y + h)
                inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
                iou = inter / (w * h + (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]) - inter + 1e-9)
                blind = not bool(((iou >= 0.5) & (det[:, 4] >= c.get("blind_score", 0.3))).any())
            centres.append((cx, cy, side, blind))
        if not centres:
            return
        add = [[] for _ in range(3)]
        occupied = []
        for cx, cy, side, blind in centres:
            k = 0 if side < 32 else (1 if side < 64 else 2)
            s = self.strides[k]
            H, W = ch["shapes"][k]
            rho = max(0.5 * s, 0.35 * side)
            js = np.arange(max(int((cx - rho) // s), 0), min(int((cx + rho) // s) + 1, W))
            is_ = np.arange(max(int((cy - rho) // s), 0), min(int((cy + rho) // s) + 1, H))
            cells = [(i, j) for i in is_ for j in js if (j + 0.5) * s - cx >= -rho and (j + 0.5) * s - cx <= rho and (i + 0.5) * s - cy >= -rho and (i + 0.5) * s - cy <= rho]
            if not cells:
                cells = [(min(int(cy // s), H - 1), min(int(cx // s), W - 1))]
            wgt = c.get("w_blind", 1.0) if blind else 1.0
            for i, j in cells:
                add[k].append((i, j, 1.0, wgt))
            self.pos_cells[k] += len(cells)
            self.stats["pos"] += len(cells)
            self.stats["blind_pos"] += len(cells) if blind else 0
            occupied.append((cx, cy, max(side, 2 * self.strides[k])))
        occ = np.array(occupied)

        def away(cx_, cy_):
            return bool(np.all(np.hypot(occ[:, 0] - cx_, occ[:, 1] - cy_) > 1.5 * occ[:, 2]))

        # hard negatives: the highest-scoring cells of every level that are away from every labelled row
        pool = []
        for k in range(3):
            sc = ch["cell_score"][k]
            H, W = sc.shape
            flat = sc.flatten()
            top = torch.topk(flat, min(c.get("n_hard", 32) * 4, flat.numel()))
            for v, idx in zip(top.values.cpu().numpy(), top.indices.cpu().numpy()):
                if v < c.get("hard_score", 0.1):
                    break
                i, j = divmod(int(idx), W)
                if away((j + 0.5) * self.strides[k], (i + 0.5) * self.strides[k]):
                    pool.append((float(v), k, i, j))
        pool.sort(reverse=True)
        for v, k, i, j in pool[:c.get("n_hard", 32)]:
            add[k].append((i, j, 0.0, c.get("w_hard", 1.0)))
            self.stats["hard_neg"] += 1
        # random negatives (level 0 mostly: the small targets live there)
        H0, W0 = ch["shapes"][0]
        got, tries = 0, 0
        while got < c.get("n_rand", 128) and tries < 20 * c.get("n_rand", 128):
            tries += 1
            i, j = int(self.rng.integers(H0)), int(self.rng.integers(W0))
            if away((j + 0.5) * self.strides[0], (i + 0.5) * self.strides[0]):
                add[0].append((i, j, 0.0, c.get("w_rand", 1.0)))
                got += 1
        self.stats["rand_neg"] += got
        for k in range(3):
            if not add[k]:
                continue
            ii = torch.tensor([a[0] for a in add[k]], device=self.device)
            jj = torch.tensor([a[1] for a in add[k]], device=self.device)
            feat = regf[k][0][:, ii, jj].T.half()
            self.buf[k]["x"].append(feat)
            self.buf[k]["y"].append(torch.tensor([a[2] for a in add[k]], device=self.device))
            self.buf[k]["w"].append(torch.tensor([a[3] for a in add[k]], device=self.device))
            cap = c.get("buffer", 4000)
            x = torch.cat(self.buf[k]["x"])
            if len(x) > cap:  # keep the most recent samples
                for key in ("x", "y", "w"):
                    t = torch.cat(self.buf[k][key])[-cap:]
                    self.buf[k][key] = [t]

    def adapt(self, rows, det):
        """rows: [(tlwh box in image pixels, number of frames the track has been written)] of the frame just tracked; det: its detections."""
        self.frame += 1
        c = self.cfg
        if self.opt is None:
            return
        self._collect(rows, det)
        if self.frame < c.get("warmup", 3):
            return
        for _ in range(c.get("steps", 1)):
            loss, n = 0.0, 0
            for k in range(3):
                if not self.buf[k]["x"]:
                    continue
                x, y, w = (torch.cat(self.buf[k][key]) for key in ("x", "y", "w"))
                if y.sum() < 1:
                    continue
                m = c.get("batch", 256)
                pos, neg = torch.nonzero(y > 0.5)[:, 0], torch.nonzero(y < 0.5)[:, 0]
                pi = pos[torch.randint(len(pos), (min(m // 2, max(len(pos), 1)),), device=self.device)]
                ni = neg[torch.randint(len(neg), (min(m // 2, max(len(neg), 1)),), device=self.device)] if len(neg) else neg
                idx = torch.cat([pi, ni])
                H, Wd = self.cache["shapes"][k]
                pi = max(self.pos_cells[k], 1) / (self.frame * H * Wd)  # natural share of positive cells at this level
                shift = float(np.log((1 - pi) / pi)) if c.get("prior_correction", True) else 0.0  # batches are 50 / 50: logit_balanced = logit_natural + shift
                logit = x[idx].float() @ self.W[k].T + self.b[k] + shift
                loss = loss + (F.binary_cross_entropy_with_logits(logit[:, 0], y[idx], weight=w[idx], reduction="sum") / len(idx))
                n += 1
            if n == 0:
                return
            reg = sum(((self.W[k] - self.W0[k]) ** 2).sum() + ((self.b[k] - self.b0[k]) ** 2).sum() for k in range(3))
            total = loss + c.get("lam", 1.0) * reg
            self.opt.zero_grad()
            total.backward()
            self.opt.step()
            self.stats["updates"] += 1
