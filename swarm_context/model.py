"""Relational transformer over the boxes of a causal window (Exp066).

Every box of the window is a token. A token may attend to tokens of its own frame and of earlier frames. The geometry
between two boxes enters as an edge: it biases the attention and is added to the message. The output is a correction
to the detector's own logit, so an untrained model returns the detector's score.

To keep the model from memorising a video's layout, a box does not get its absolute position or size (node_abs=False)
and sees the boxes outside its own neighbourhood only through distance, size ratio, time and score, not through their
direction (peer_dir=False); during training a share of the other boxes is hidden at random (key_drop).

With app > 0 every box also carries the detector's feature at its centre (app channels). It is used only through
similarities under a learned embedding: how much the box looks like the confident boxes (score >= app_thr) outside its
own neighbourhood (the other members of the swarm: "peer") and like the confident boxes inside it (its own earlier
boxes: "hist"). The raw feature never enters the model, so it cannot recognise the scene, and the similarities do not
pass through the attention layers: a small separate head turns them into a second correction that is added to the
output. app_mode "other" replaces the swarm's own members by the confident boxes of another sample of the batch
(control: generic targets, not this swarm).

scope selects which other tokens a token may see:
  all    every box of the window (swarm context)
  tube   only boxes close to it (distance <= r0 + r1 * frames apart, in its own box sides): its own history and duplicates
  peers  only boxes outside that neighbourhood, plus itself
  self   nothing but itself (no context: a per-box recalibration of the score)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NODE, EDGE = 9, 11


def logit(p):
    p = p.clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p))


def features(box, step, wh, k):
    """box [B,N,5] x,y,w,h,score in pixels; step [B,N] frames relative to the last frame (<= 0); wh [B,2]."""
    w, h = box[..., 2].clamp(min=1e-3), box[..., 3].clamp(min=1e-3)
    cx, cy = box[..., 0] + w / 2, box[..., 1] + h / 2
    side = (w * h).sqrt()
    W, H = wh[:, :1], wh[:, 1:]
    lg = logit(box[..., 4])
    border = torch.minimum(torch.minimum(cx, W - cx), torch.minimum(cy, H - cy)) / side  # distance to the image border in box sides
    node = torch.stack([lg / 5, box[..., 4], torch.log(w / W) / 4 + 1, torch.log(h / H) / 4 + 1, torch.log(w / h), cx / W, cy / H,
                        torch.asinh(border.clamp(min=-2)) / 3, step / max(k, 1)], -1)
    dx = (cx[:, None, :] - cx[:, :, None]) / side[:, :, None]  # [B, query, key]
    dy = (cy[:, None, :] - cy[:, :, None]) / side[:, :, None]
    dist = (dx * dx + dy * dy).sqrt()
    dt = (step[:, None, :] - step[:, :, None]).float()
    x1, y1, x2, y2 = box[..., 0], box[..., 1], box[..., 0] + w, box[..., 1] + h
    iw = (torch.minimum(x2[:, None, :], x2[:, :, None]) - torch.maximum(x1[:, None, :], x1[:, :, None])).clamp(min=0)
    ih = (torch.minimum(y2[:, None, :], y2[:, :, None]) - torch.maximum(y1[:, None, :], y1[:, :, None])).clamp(min=0)
    area = w * h
    iou = iw * ih / (area[:, None, :] + area[:, :, None] - iw * ih)
    edge = torch.stack([torch.asinh(dx), torch.asinh(dy), torch.asinh(dist), dx.clamp(-3, 3) / 3, dy.clamp(-3, 3) / 3,
                        torch.log(w[:, None, :] / w[:, :, None]).clamp(-3, 3), torch.log(h[:, None, :] / h[:, :, None]).clamp(-3, 3),
                        dt / max(k, 1), (dt == 0).float(), iou, (lg[:, None, :] - lg[:, :, None]) / 5], -1)
    return node, edge, dist, dt, lg


def sim_stats(sim, pair):
    """sim, pair [B,N,M]. Best, mean of the three best and mean similarity over the allowed references, and whether there is one."""
    n = pair.sum(-1)
    has = (n > 0).float()
    top = sim.masked_fill(~pair, -2.0).topk(min(3, sim.shape[-1]), dim=-1).values
    ok = top > -2.0
    top3 = (top * ok).sum(-1) / ok.sum(-1).clamp(min=1)
    mean = (sim * pair).sum(-1) / n.clamp(min=1)
    return torch.stack([top[..., 0] * has, top3 * has, mean * has, has], -1)


class Layer(nn.Module):
    def __init__(self, d, de, heads, drop):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.eb, self.ev = nn.Linear(de, heads), nn.Linear(de, d)
        self.o = nn.Linear(d, d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, e, allow):
        b, n, d = x.shape
        y = self.n1(x)
        q = self.q(y).view(b, n, self.h, self.dk).transpose(1, 2)
        k = self.k(y).view(b, n, self.h, self.dk).transpose(1, 2)
        v = self.v(y).view(b, n, self.h, self.dk).transpose(1, 2)
        att = q @ k.transpose(-1, -2) / self.dk ** 0.5 + self.eb(e).permute(0, 3, 1, 2)
        att = att.masked_fill(~allow[:, None], float("-inf")).softmax(-1)
        att = self.drop(att)
        msg = (att @ v).transpose(1, 2).reshape(b, n, d)
        # what the attended boxes look like relative to this one; the edges are averaged before the projection to keep memory low
        msg = msg + self.ev(torch.einsum("bqk,bqke->bqe", att.mean(1), e))
        x = x + self.drop(self.o(msg))
        return x + self.drop(self.ff(self.n2(x)))


class SwarmContext(nn.Module):
    def __init__(self, k=8, d=96, de=32, heads=4, layers=3, drop=0.1, scope="all", r0=2.0, r1=1.0, node_abs=True, peer_dir=True, key_drop=0.0,
                 app=0, app_mode="peer+hist", app_e=64, app_thr=0.6, app_drop=0.3):
        super().__init__()
        self.cfg = dict(k=k, d=d, de=de, heads=heads, layers=layers, drop=drop, scope=scope, r0=r0, r1=r1, node_abs=node_abs, peer_dir=peer_dir, key_drop=key_drop,
                        app=app, app_mode=app_mode, app_e=app_e, app_thr=app_thr, app_drop=app_drop)
        self.app, self.app_mode, self.app_thr = app, app_mode, app_thr
        if app:
            self.embed = nn.Sequential(nn.Dropout(app_drop), nn.Linear(app, app_e))
            self.app_head = nn.Sequential(nn.Linear(9, 32), nn.GELU(), nn.Linear(32, 1))
            nn.init.zeros_(self.app_head[-1].weight)
            nn.init.zeros_(self.app_head[-1].bias)
            self.register_buffer("app_mu", torch.zeros(app))
            self.register_buffer("app_sd", torch.ones(app))
        self.k, self.scope, self.r0, self.r1 = k, scope, r0, r1
        self.node_abs, self.peer_dir, self.key_drop = node_abs, peer_dir, key_drop
        self.node = nn.Sequential(nn.Linear(NODE, d), nn.GELU(), nn.Linear(d, d))
        self.edge = nn.Sequential(nn.Linear(EDGE, de), nn.GELU(), nn.Linear(de, de))
        self.layers = nn.ModuleList([Layer(d, de, heads, drop) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def appearance(self, feat, score, valid, near, dt, eye):
        z = F.normalize(self.embed((feat - self.app_mu) / self.app_sd), dim=-1)
        conf = ((score >= self.app_thr) & valid)[:, None, :] & (dt <= 0) & ~eye
        sim = z @ z.transpose(1, 2)
        zero = torch.zeros(*sim.shape[:2], 4, device=sim.device)
        if self.app_mode == "other":
            other = ((score >= self.app_thr) & valid).roll(1, 0)[:, None, :].expand(-1, sim.shape[1], -1)
            return torch.cat([sim_stats(z @ z.roll(1, 0).transpose(1, 2), other), zero], -1)
        peer = sim_stats(sim, conf & ~near) if "peer" in self.app_mode else zero
        hist = sim_stats(sim, conf & near) if "hist" in self.app_mode else zero
        return torch.cat([peer, hist], -1)

    def forward(self, box, step, wh, valid, feat=None):
        node, edge, dist, dt, lg = features(box, step, wh, self.k)
        n = box.shape[1]
        eye = torch.eye(n, dtype=torch.bool, device=box.device)[None]
        allow = valid[:, None, :] & (dt <= 0)
        near = dist <= self.r0 + self.r1 * dt.abs()
        if self.scope == "tube":
            allow = allow & near
        elif self.scope == "peers":
            allow = allow & ~near
        elif self.scope == "self":
            allow = allow & eye
        if self.training and self.key_drop > 0:
            allow = allow & (torch.rand_like(dist) >= self.key_drop)
        allow = allow | eye
        if not self.node_abs:  # no absolute size or position
            node = node * node.new_tensor([1, 1, 0, 0, 1, 0, 0, 1, 1])
        if not self.peer_dir:  # outside the neighbourhood only the distance is kept, not the direction
            keep = near.unsqueeze(-1) | edge.new_tensor([0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1]).bool()
            edge = edge * keep
        x, e = self.node(node), self.edge(edge)
        for layer in self.layers:
            x = layer(x, e, allow)
        out = lg + self.out(x).squeeze(-1)
        if self.app:
            out = out + self.app_head(torch.cat([lg[..., None] / 5, self.appearance(feat, box[..., 4], valid, near, dt, eye)], -1)).squeeze(-1)
        return out
