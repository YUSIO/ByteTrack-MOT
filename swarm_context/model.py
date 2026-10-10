"""Relational transformer over the boxes of a causal window (Exp066).

Every box of the window is a token. A token may attend to tokens of its own frame and of earlier frames. The geometry
between two boxes enters as an edge: it biases the attention and is added to the message. The output is a correction
to the detector's own logit, so an untrained model returns the detector's score.

To keep the model from memorising a video's layout, a box does not get its absolute position or size (node_abs=False)
and sees the boxes outside its own neighbourhood only through distance, size ratio, time and score, not through their
direction (peer_dir=False); during training a share of the other boxes is hidden at random (key_drop).

scope selects which other tokens a token may see:
  all    every box of the window (swarm context)
  tube   only boxes close to it (distance <= r0 + r1 * frames apart, in its own box sides): its own history and duplicates
  peers  only boxes outside that neighbourhood, plus itself
"""
import torch
import torch.nn as nn

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
    def __init__(self, k=8, d=96, de=32, heads=4, layers=3, drop=0.1, scope="all", r0=2.0, r1=1.0, node_abs=True, peer_dir=True, key_drop=0.0):
        super().__init__()
        self.cfg = dict(k=k, d=d, de=de, heads=heads, layers=layers, drop=drop, scope=scope, r0=r0, r1=r1, node_abs=node_abs, peer_dir=peer_dir, key_drop=key_drop)
        self.k, self.scope, self.r0, self.r1 = k, scope, r0, r1
        self.node_abs, self.peer_dir, self.key_drop = node_abs, peer_dir, key_drop
        self.node = nn.Sequential(nn.Linear(NODE, d), nn.GELU(), nn.Linear(d, d))
        self.edge = nn.Sequential(nn.Linear(EDGE, de), nn.GELU(), nn.Linear(de, de))
        self.layers = nn.ModuleList([Layer(d, de, heads, drop) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, box, step, wh, valid):
        node, edge, dist, dt, lg = features(box, step, wh, self.k)
        n = box.shape[1]
        eye = torch.eye(n, dtype=torch.bool, device=box.device)[None]
        allow = valid[:, None, :] & (dt <= 0)
        near = dist <= self.r0 + self.r1 * dt.abs()
        if self.scope == "tube":
            allow = allow & near
        elif self.scope == "peers":
            allow = allow & ~near
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
        return lg + self.out(x).squeeze(-1)
