"""Track-supervised online score adapter: a pluggable component for YOLO-type (anchor-free, dense) detectors.

What it needs from a detector: the final 1 x 1 convolution of every FPN level that turns a feature map into the per-cell target
score (YOLOX: `head.obj_preds[k]`; YOLOv8 / 11 / 26: the last conv of `Detect.cv3[k]`, class channel 0 for a single-class
detector). It attaches a forward hook to each of these layers and:
  * records the layer's input (the cell features) of the current frame;
  * replaces the layer's output on the target channel by the adapted logit  w_k . feature + b_k  where the source logit already
    says `sigmoid(source) >= gate` (cells the source detector finds plausible), and keeps the source logit elsewhere. The
    detector's boxes, classes, NMS and decoding are untouched; only the target score moves.
After each tracked frame `update(rows, dets, geometry)` turns the rows the tracker wrote into labelled cells (positive = the cell
holding a written row's centre, negative = the highest-scoring cells and random cells away from every row), appends them to a
per-level replay buffer and takes a few Adam steps on the 129 weights per level, starting from the source weights, with an
optional L2-SP pull towards them. The batch is 50 / 50 positive / negative; the logit is shifted by the natural positive share
of cells (log((1 - pi) / pi)) during training so that the deployed logit stays calibrated. Causal: frame t's rows only change
the scores of frames > t. `begin_sequence()` resets everything to the source weights.

Interface used by a detector wrapper:  adapter = OnlineScoreAdapter(layers, cfg); adapter.attach();
  per sequence: adapter.begin_sequence();  per frame: dets = detector(img)  (hooks active);  rows = tracker(dets);
  adapter.update(rows, dets, geometry)   with geometry = {"scale": s, "pad": (px, py)}  (original pixel -> detector input pixel:
  x_in = s * x + px). Nothing here depends on the tracker or on the detector family.
"""
import numpy as np
import torch
import torch.nn.functional as F


class OnlineScoreAdapter:
    def __init__(self, layers, cfg, channel=0):
        self.layers, self.cfg, self.channel = list(layers), dict(cfg), channel
        self.L = len(self.layers)
        self.W0 = [l.weight[channel, :, 0, 0].detach().clone() for l in self.layers]
        self.b0 = [l.bias[channel].detach().clone() for l in self.layers]
        self.handles = []
        self.feat = [None] * self.L
        self.score = [None] * self.L
        self.begin_sequence()

    # ------------------------------------------------------------------ hooks
    def attach(self):
        for k, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_hook(self._make_hook(k)))
        return self

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _make_hook(self, k):
        def hook(module, inputs, output):
            feat = inputs[0]
            self.feat[k] = feat
            src = output[:, self.channel]
            c = self.cfg
            if c.get("lr", 0.0) > 0:
                adapted = torch.einsum("bchw,c->bhw", feat.float(), self.W[k].detach().to(feat.device)) + self.b[k].detach().to(feat.device)
                gate = c.get("gate", 0.0)
                new = torch.where(src.sigmoid() >= gate, adapted, src.float()) if gate > 0 else adapted
                out = output.clone()
                out[:, self.channel] = new.to(out.dtype)
                self.score[k] = new.sigmoid()[0]
                return out
            self.score[k] = src.sigmoid()[0]
            return None
        return hook

    # ------------------------------------------------------------------ state
    def begin_sequence(self):
        dev = self.W0[0].device
        self.W = [w.clone().requires_grad_(True) for w in self.W0]
        self.b = [b.clone().requires_grad_(True) for b in self.b0]
        c = self.cfg
        self.opt = torch.optim.Adam(self.W + self.b, lr=c["lr"]) if c.get("lr", 0.0) > 0 else None
        self.buf = [{"x": [], "y": [], "w": []} for _ in range(self.L)]
        self.pos_cells = [0] * self.L
        self.frame = 0
        self.rng = np.random.default_rng(c.get("seed", 0))
        self.stats = {"updates": 0, "pos": 0, "blind_pos": 0, "hard_neg": 0, "rand_neg": 0}
        self.device = dev

    # ------------------------------------------------------------ supervision
    @torch.no_grad()
    def _collect(self, rows, dets, geo):
        c = self.cfg
        s, (px, py) = geo["scale"], geo["pad"]
        shapes = [tuple(f.shape[-2:]) for f in self.feat]
        in_w = geo.get("input_wh", (shapes[0][1] * geo.get("stride0", 8), shapes[0][0] * geo.get("stride0", 8)))[0]
        strides = [in_w / sh[1] for sh in shapes]
        centres = []
        for box, age in rows:
            if age < c.get("pos_age", 2):
                continue
            x, y, w, h = box
            if w * h <= 0:
                continue
            blind = True
            if len(dets):
                ix = np.clip(np.minimum(dets[:, 2], x + w) - np.maximum(dets[:, 0], x), 0, None)
                iy = np.clip(np.minimum(dets[:, 3], y + h) - np.maximum(dets[:, 1], y), 0, None)
                inter = ix * iy
                iou = inter / (w * h + (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1]) - inter + 1e-9)
                blind = not bool(((iou >= 0.5) & (dets[:, 4] >= c.get("blind_score", 0.3))).any())
            centres.append(((x + w / 2) * s + px, (y + h / 2) * s + py, float(np.sqrt(w * h)) * s, blind))
        if not centres:
            return
        add = [[] for _ in range(self.L)]
        occupied, row_cells = [], []
        for cx, cy, side, blind in centres:
            k = next((j for j in range(self.L) if side < 4 * strides[j]), self.L - 1)
            st = strides[k]
            H, W = shapes[k]
            rho = max(0.5 * st, 0.35 * side)
            js = np.arange(max(int((cx - rho) // st), 0), min(int((cx + rho) // st) + 1, W))
            is_ = np.arange(max(int((cy - rho) // st), 0), min(int((cy + rho) // st) + 1, H))
            cells = [(i, j) for i in is_ for j in js if abs((j + 0.5) * st - cx) <= rho and abs((i + 0.5) * st - cy) <= rho]
            if not cells:
                cells = [(min(max(int(cy // st), 0), H - 1), min(max(int(cx // st), 0), W - 1))]
            wgt = c.get("w_blind", 1.0) if blind else 1.0
            for i, j in cells:
                add[k].append((i, j, 1.0, wgt))
            row_cells.append((k, min(max(int(cy // st), 0), H - 1), min(max(int(cx // st), 0), W - 1)))
            self.pos_cells[k] += len(cells)
            self.stats["pos"] += len(cells)
            self.stats["blind_pos"] += len(cells) if blind else 0
            occupied.append((cx, cy, max(side, 2 * st)))
        # neighbour negatives (NMS-free heads keep one peak per object): the cells around a row's centre cell that are not positives
        n_nbr = c.get("n_nbr", 0)
        if n_nbr > 0:
            R = c.get("nbr_radius", 2)
            pos_set = [{(a[0], a[1]) for a in add[k] if a[2] > 0.5} for k in range(self.L)]
            for k, ic, jc in row_cells:
                H, W = shapes[k]
                ring = [(ic + di, jc + dj) for di in range(-R, R + 1) for dj in range(-R, R + 1)
                        if (di or dj) and 0 <= ic + di < H and 0 <= jc + dj < W and (ic + di, jc + dj) not in pos_set[k]]
                if len(ring) > n_nbr:
                    ring = [ring[q] for q in self.rng.choice(len(ring), n_nbr, replace=False)]
                for i, j in ring:
                    add[k].append((i, j, 0.0, c.get("w_nbr", 1.0)))
                self.stats["nbr_neg"] = self.stats.get("nbr_neg", 0) + len(ring)
        occ = np.array(occupied)
        away = lambda cx_, cy_: bool(np.all(np.hypot(occ[:, 0] - cx_, occ[:, 1] - cy_) > 1.5 * occ[:, 2]))
        pool = []
        for k in range(self.L):
            sc = self.score[k]
            if sc is None:
                continue
            H, W = sc.shape
            flat = sc.flatten()
            top = torch.topk(flat, min(c.get("n_hard", 32) * 4, flat.numel()))
            for v, idx in zip(top.values.cpu().numpy(), top.indices.cpu().numpy()):
                if v < c.get("hard_score", 0.1):
                    break
                i, j = divmod(int(idx), W)
                if away((j + 0.5) * strides[k], (i + 0.5) * strides[k]):
                    pool.append((float(v), k, i, j))
        pool.sort(reverse=True)
        for v, k, i, j in pool[:c.get("n_hard", 32)]:
            add[k].append((i, j, 0.0, c.get("w_hard", 1.0)))
            self.stats["hard_neg"] += 1
        H0, W0 = shapes[0]
        got, tries = 0, 0
        while got < c.get("n_rand", 128) and tries < 20 * c.get("n_rand", 128):
            tries += 1
            i, j = int(self.rng.integers(H0)), int(self.rng.integers(W0))
            if away((j + 0.5) * strides[0], (i + 0.5) * strides[0]):
                add[0].append((i, j, 0.0, c.get("w_rand", 1.0)))
                got += 1
        self.stats["rand_neg"] += got
        for k in range(self.L):
            if not add[k]:
                continue
            ii = torch.tensor([a[0] for a in add[k]], device=self.device)
            jj = torch.tensor([a[1] for a in add[k]], device=self.device)
            feat = self.feat[k][0][:, ii, jj].T.half().clone()  # clone: a normal tensor even if the detector ran in inference mode
            self.buf[k]["x"].append(feat)
            self.buf[k]["y"].append(torch.tensor([a[2] for a in add[k]], device=self.device))
            self.buf[k]["w"].append(torch.tensor([a[3] for a in add[k]], device=self.device))
            cap = c.get("buffer", 4000)
            if sum(len(t) for t in self.buf[k]["x"]) > cap:
                for key in ("x", "y", "w"):
                    self.buf[k][key] = [torch.cat(self.buf[k][key])[-cap:]]

    def update(self, rows, dets, geometry):
        """rows: [(tlwh box in image pixels, frames the track has been written)] of the frame just tracked; dets: its detections (x1, y1, x2, y2, score)."""
        self.frame += 1
        c = self.cfg
        if self.opt is None or self.feat[0] is None:
            return
        self._collect(rows, dets, geometry)
        if self.frame < c.get("warmup", 3):
            return
        shapes = [tuple(f.shape[-2:]) for f in self.feat]
        for _ in range(c.get("steps", 1)):
            loss, n = 0.0, 0
            for k in range(self.L):
                if not self.buf[k]["x"]:
                    continue
                x, y, w = (torch.cat(self.buf[k][key]) for key in ("x", "y", "w"))
                if y.sum() < 1:
                    continue
                m = c.get("batch", 256)
                pos, neg = torch.nonzero(y > 0.5)[:, 0], torch.nonzero(y < 0.5)[:, 0]
                pi_ = pos[torch.randint(len(pos), (min(m // 2, max(len(pos), 1)),), device=self.device)]
                ni_ = neg[torch.randint(len(neg), (min(m // 2, max(len(neg), 1)),), device=self.device)] if len(neg) else neg
                idx = torch.cat([pi_, ni_])
                pi = max(self.pos_cells[k], 1) / (self.frame * shapes[k][0] * shapes[k][1])
                shift = float(np.log((1 - pi) / pi)) if c.get("prior_correction", True) else 0.0
                logit = x[idx].float() @ self.W[k] + self.b[k] + shift
                loss = loss + F.binary_cross_entropy_with_logits(logit, y[idx], weight=w[idx], reduction="sum") / len(idx)
                n += 1
            if n == 0:
                return
            reg = sum(((self.W[k] - self.W0[k]) ** 2).sum() + ((self.b[k] - self.b0[k]) ** 2).sum() for k in range(self.L))
            total = loss + c.get("lam", 0.0) * reg
            self.opt.zero_grad()
            total.backward()
            self.opt.step()
            self.stats["updates"] += 1
