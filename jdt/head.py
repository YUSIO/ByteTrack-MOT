"""Track head on the detector's stride-8 features, with switchable geometry / formation / motion inputs (Exp071).

For one track (a query) the head looks at a square window of the current frame's P3 feature around a placement
position and answers: is this track's UAV inside the window (exist), where is its centre (a heat map over a G x G
grid) and what is its box (offset and size regressed at that cell). Inputs, all in network-input pixels:

  always    P3 of the window (roi_align, G x G), the detector's own stride-8 objectness in the window, and the
            relative coordinates of the grid (the placement is the window centre).
  tmpl      detection features as an association cue: P3 pooled at the track's last detector box in the frame of
            that box (a 4 x 4 roi), as a FiLM vector, and its cosine similarity to every cell of the window.
  mot       motion: a Gaussian at the track's own constant-velocity prediction; vector (vx, vy) / side, gap / 5.
  form      formation: Gaussians at what every formation neighbour says (neighbour's current centre + the offset it
            had to this track in the reference frame); Gaussians at the centres of the other members that are
            tracked in this frame ("taken"); vector (neighbour count / 6, spread of the neighbour estimates / side).
  geo       geometry: vector of log(side / 20), log(w / h), the centre in image fractions, the distance to the image
            border in sides, log(median member side / side).

A group that is switched off is fed as zeros; every head has the same structure and parameter count, so heads that
differ only in their groups are trained side by side on the same samples.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

G = 12            # window grid
TG = 4            # template grid
STRIDE = 8
GROUPS = ("tmpl", "mot", "form", "geo")
MAX_NB, MAX_OCC = 6, 16


def window_features(p3, obj_logit, qimg, centre, half, tmpl_feat_src=None, tbox=None, timg=None):
    """roi_align of the window (and optionally the template) for every query.

    p3: B x C x h x w; obj_logit: B x 1 x h x w; qimg: Q (batch index of the search image); centre: Q x 2; half: Q.
    tbox: Q x 4 (x1, y1, x2, y2) on the template image `timg` of `tmpl_feat_src`.
    """
    rois = torch.cat((qimg[:, None].float(), centre - half[:, None], centre + half[:, None]), 1).to(p3.dtype)
    fs = roi_align(p3, rois, G, 1.0 / STRIDE, sampling_ratio=2, aligned=True)
    ob = roi_align(torch.sigmoid(obj_logit.float()).to(p3.dtype), rois, G, 1.0 / STRIDE, sampling_ratio=2, aligned=True)
    tf = None
    if tbox is not None:
        trois = torch.cat((timg[:, None].float(), tbox), 1).to(tmpl_feat_src.dtype)
        tf = roi_align(tmpl_feat_src, trois, TG, 1.0 / STRIDE, sampling_ratio=2, aligned=True)
    return fs.float(), ob.float(), None if tf is None else tf.float()


def grid_points(centre, half):
    """Q x G*G x 2 cell centres of every window, and the cell step Q."""
    step = 2 * half / G
    i = torch.arange(G, device=centre.device, dtype=centre.dtype) + 0.5
    u = centre[:, 0:1] - half[:, None] + i[None] * step[:, None]  # Q x G (x)
    v = centre[:, 1:2] - half[:, None] + i[None] * step[:, None]  # Q x G (y)
    pts = torch.stack((u[:, None, :].expand(-1, G, -1), v[:, :, None].expand(-1, -1, G)), -1)  # Q x G(y) x G(x) x 2
    return pts.reshape(len(centre), G * G, 2), step


def gauss_max(pts, P, mask, sigma):
    """max over points P (Q x M x 2, mask Q x M) of exp(-d^2 / 2 sigma^2) on the grid pts (Q x N x 2) -> Q x 1 x G x G."""
    if P.shape[1] == 0:
        return pts.new_zeros(len(pts), 1, G, G)
    d2 = ((pts[:, :, None, :] - P[:, None, :, :]) ** 2).sum(-1)
    g = torch.exp(-d2 / (2 * sigma[:, None, None] ** 2)) * mask[:, None, :].float()
    return g.amax(2).view(len(pts), 1, G, G)


class TrackHead(nn.Module):
    def __init__(self, groups, channels=128, width=64):
        super().__init__()
        self.groups = tuple(groups)
        assert all(g in GROUPS for g in self.groups), groups
        self.reduce = nn.Sequential(nn.Conv2d(channels, width, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.tvec = nn.Sequential(nn.Linear(2 * channels, width), nn.ReLU(inplace=True))
        nvec = width + 3 + 2 + 6
        self.vmlp = nn.Sequential(nn.Linear(nvec, 128), nn.ReLU(inplace=True), nn.Linear(128, 128), nn.ReLU(inplace=True))
        self.film = nn.Linear(128, 2 * width)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        cin = width + 1 + 1 + 1 + 2 + 2  # P3, objectness, correlation, motion, formation estimate + taken, coordinates
        self.c1 = nn.Sequential(nn.Conv2d(cin, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(nn.Conv2d(width, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
                                nn.Conv2d(width, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.heat = nn.Conv2d(width, 1, 1)
        self.box = nn.Conv2d(width, 4, 1)
        self.exist = nn.Sequential(nn.Linear(2 * width + 128, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def on(self, g):
        return 1.0 if g in self.groups else 0.0

    def forward(self, fs, ob, tf, q):
        """fs: Q x C x G x G; ob: Q x 1 x G x G; tf: Q x C x TG x TG; q: dict of query tensors (see `inputs`)."""
        Q = len(fs)
        pts, step = grid_points(q["centre"], q["half"])
        side = q["side"]
        t_on, m_on, f_on, g_on = (self.on(g) for g in GROUPS)
        tmean = tf.mean((2, 3))
        tvec = self.tvec(torch.cat((tmean, tf.amax((2, 3))), 1)) * t_on
        corr = (F.normalize(fs, dim=1) * F.normalize(tmean, dim=1)[:, :, None, None]).sum(1, keepdim=True) * t_on
        mot = gauss_max(pts, q["cv"][:, None], torch.ones(Q, 1, device=fs.device), 0.5 * side * (1 + q["gap"] / 5)) * m_on
        fest = gauss_max(pts, q["nb"], q["nb_mask"], 0.5 * side) * f_on
        occ = gauss_max(pts, q["occ"], q["occ_mask"], 0.35 * side) * f_on
        coord = ((pts - q["centre"][:, None]) / q["half"][:, None, None]).permute(0, 2, 1).reshape(Q, 2, G, G)
        x = torch.cat((self.reduce(fs), ob, corr, mot, fest, occ, coord), 1)
        mvec = torch.stack((q["vel"][:, 0] / side, q["vel"][:, 1] / side, q["gap"] / 5), 1) * m_on
        fvec = torch.stack((q["nb_mask"].sum(1) / MAX_NB, q["spread"].clamp(max=5) / 5), 1) * f_on
        v = self.vmlp(torch.cat((tvec, mvec, fvec, q["geo"] * g_on), 1))
        gamma, beta = self.film(v).chunk(2, 1)
        h = self.c1(x)
        h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        h = self.c2(h)
        heat = self.heat(h).view(Q, G * G)
        box = self.box(h).view(Q, 4, G * G)
        exist = self.exist(torch.cat((h.mean((2, 3)), h.amax((2, 3)), v), 1))[:, 0]
        return exist, heat, box, pts, step


def targets(pts, step, q):
    """Target cell and box offsets of the positive queries (exist == 1)."""
    gt = q["gt"]  # Q x 4 (cx, cy, w, h)
    lo = q["centre"] - q["half"][:, None]
    cell = ((gt[:, :2] - lo) / step[:, None]).floor().clamp(0, G - 1).long()
    idx = cell[:, 1] * G + cell[:, 0]
    c = pts[torch.arange(len(pts)), idx]
    off = torch.stack(((gt[:, 0] - c[:, 0]) / step, (gt[:, 1] - c[:, 1]) / step,
                       torch.log(gt[:, 2] / q["wh"][:, 0]), torch.log(gt[:, 3] / q["wh"][:, 1])), 1)
    return idx, off


def loss(out, q):
    exist, heat, box, pts, step = out
    y = q["exist"]
    l_exist = F.binary_cross_entropy_with_logits(exist, y)
    pos = y > 0.5
    if not pos.any():
        return l_exist, {"exist": l_exist.item(), "heat": 0.0, "box": 0.0}
    idx, off = targets(pts[pos], step[pos], {k: v[pos] for k, v in q.items()})
    l_heat = F.cross_entropy(heat[pos], idx)
    pred = box[pos][torch.arange(int(pos.sum())), :, idx]
    l_box = F.smooth_l1_loss(pred, off, beta=0.1)
    return l_exist + l_heat + 2.0 * l_box, {"exist": l_exist.item(), "heat": l_heat.item(), "box": l_box.item()}


def decode_box(out, q):
    """Presence probability and the box (x1, y1, w, h in network-input pixels) at the arg-max cell."""
    exist, heat, box, pts, step = out
    idx = heat.argmax(1)
    ar = torch.arange(len(idx), device=idx.device)
    c = pts[ar, idx]
    b = box[ar, :, idx]
    cx, cy = c[:, 0] + b[:, 0] * step, c[:, 1] + b[:, 1] * step
    w, h = q["wh"][:, 0] * torch.exp(b[:, 2].clamp(-2, 2)), q["wh"][:, 1] * torch.exp(b[:, 3].clamp(-2, 2))
    return torch.sigmoid(exist), torch.stack((cx - w / 2, cy - h / 2, w, h), 1)
