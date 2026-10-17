"""STRN (Exp086): swarm-conditioned track re-observation network.

One query per lost-track state. Inputs (side = sqrt(w h) of the track's last box; the proposal is the estimated centre
with the last box size):
  search crops   frames t and t-1, a window of SEARCH_ZOOM sides around the proposal centre, SEARCH_IN x SEARCH_IN
  template crop  the frame of the last real observation, TMPL_ZOOM sides around that box, TMPL_IN x TMPL_IN
  candidates     detector boxes (score > 0.001) whose centre is inside the search window, top N_CAND by score:
                 (dx, dy) / side, log(w / side), log(h / side), score, max IoU with a tracked box
  neighbours     up to N_NB members (same selection as the formation estimate): (c0 - p0_j) / side, (p_j - p0_j) / side,
                 (p_j + c0 - p0_j - proposal centre) / side
  state          gap / 30, since_real / 30, log(1 + obs), kalman std x / side, std y / side, log(side / 20),
                 log(image width / 1000), formation used
Outputs: existence logit, regressed box relative to the proposal, pointer logits over the candidates plus "use the
regressed box". `inputs()` is shared by training (from logged states) and inference, so both see the same features.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fctrack.verifier import SeedVerifier

SEARCH_ZOOM, SEARCH_IN, TMPL_ZOOM, TMPL_IN = 6.0, 128, 3.0, 64
N_CAND, N_NB, D = 16, 6, 128


def crop(img, cx, cy, side, zoom, size):
    """size x size crop covering zoom x side native pixels around (cx, cy); uint8 HWC BGR (as cv2 reads it)."""
    import cv2
    s = size / (zoom * side)
    M = np.array([[s, 0, size / 2 - s * cx], [0, s, size / 2 - s * cy]], np.float32)
    return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR if s >= 1 else cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def _iou(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, :2] + a[:, None, 2:], b[None, :, :2] + b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    return inter / np.maximum(np.prod(a[:, 2:], 1)[:, None] + np.prod(b[:, 2:], 1)[None, :] - inter, 1e-9)


def inputs(s, img_t, img_p, img_r, det, tracked):
    """One state -> numpy arrays. s: dict with prop (tlwh), rbox (tlwh), gap, since_real, obs, cov (2), formation,
    width, c0 (2), nb (list of [p0x, p0y, p1x, p1y]); det: n x 5 (x1, y1, x2, y2, score) of frame t; tracked: m x 4 tlwh."""
    prop = np.asarray(s["prop"], float)
    side = float(np.sqrt(max(prop[2] * prop[3], 1.0)))
    cx, cy = prop[0] + prop[2] / 2, prop[1] + prop[3] / 2
    rb = np.asarray(s["rbox"], float)
    rside = float(np.sqrt(max(rb[2] * rb[3], 1.0)))
    srch = np.stack([crop(img_t, cx, cy, side, SEARCH_ZOOM, SEARCH_IN), crop(img_p, cx, cy, side, SEARCH_ZOOM, SEARCH_IN)])
    tmpl = crop(img_r, rb[0] + rb[2] / 2, rb[1] + rb[3] / 2, rside, TMPL_ZOOM, TMPL_IN)
    cand, cbox = np.zeros((N_CAND, 6), np.float32), np.zeros((N_CAND, 4), np.float32)
    cmask = np.zeros(N_CAND, bool)
    if len(det):
        tl = np.column_stack((det[:, :2], det[:, 2:4] - det[:, :2]))
        c = tl[:, :2] + tl[:, 2:] / 2
        half = SEARCH_ZOOM / 2 * side
        ok = (np.abs(c[:, 0] - cx) <= half) & (np.abs(c[:, 1] - cy) <= half) & (det[:, 4] > 0.001) & (tl[:, 2] > 1) & (tl[:, 3] > 1)
        idx = np.where(ok)[0]
        idx = idx[np.argsort(-det[idx, 4])][:N_CAND]
        if len(idx):
            ov = _iou(tl[idx], tracked).max(1) if len(tracked) else np.zeros(len(idx))
            cand[:len(idx)] = np.column_stack(((c[idx, 0] - cx) / side, (c[idx, 1] - cy) / side, np.log(tl[idx, 2] / side),
                                               np.log(tl[idx, 3] / side), det[idx, 4], ov))
            cbox[:len(idx)] = tl[idx]
            cmask[:len(idx)] = True
    nb, nmask = np.zeros((N_NB, 6), np.float32), np.zeros(N_NB, bool)
    c0 = np.asarray(s["c0"], float)
    for k, (p0x, p0y, p1x, p1y) in enumerate(s["nb"][:N_NB]):
        off = c0 - (p0x, p0y)
        nb[k] = [off[0] / side, off[1] / side, (p1x - p0x) / side, (p1y - p0y) / side,
                 (p1x + off[0] - cx) / side, (p1y + off[1] - cy) / side]
        nmask[k] = True
    st = np.array([s["gap"] / 30, s["since_real"] / 30, np.log1p(s["obs"]), np.sqrt(max(s["cov"][0], 0)) / side,
                   np.sqrt(max(s["cov"][1], 0)) / side, np.log(side / 20), np.log(s["width"] / 1000), float(s["formation"])], np.float32)
    return {"srch": srch, "tmpl": tmpl, "cand": cand, "cmask": cmask, "cbox": cbox, "nb": nb, "nmask": nmask, "state": st,
            "side": np.float32(side), "centre": np.array([cx, cy], np.float32)}


def flip(x):
    """Horizontal flip of one sample around the proposal centre (crops mirrored, x components negated)."""
    y = dict(x)
    y["srch"], y["tmpl"] = x["srch"][:, :, ::-1].copy(), x["tmpl"][:, ::-1].copy()
    c, n = x["cand"].copy(), x["nb"].copy()
    c[:, 0] *= -1
    n[:, [0, 2, 4]] *= -1
    y["cand"], y["nb"] = c, n
    if "box_t" in x:
        b = x["box_t"].copy(); b[0] *= -1; y["box_t"] = b
    return y


def to_tensor(img):
    """uint8 BGR (..., H, W, 3) -> float RGB (..., 3, H, W) in [0, 1]."""
    t = torch.as_tensor(img[..., ::-1].copy() if isinstance(img, np.ndarray) else img).float() / 255.0
    return t.movedim(-1, -3)


class MLP(nn.Sequential):
    def __init__(self, i, o=D):
        super().__init__(nn.Linear(i, o), nn.ReLU(inplace=True), nn.Linear(o, o))


class STRN(nn.Module):
    def __init__(self, use_prev=True, use_nb=True, use_cand=True):
        super().__init__()
        self.use_prev, self.use_nb, self.use_cand = use_prev, use_nb, use_cand
        self.body = SeedVerifier().body
        self.proj = nn.Conv2d(256, D, 1)
        self.tproj = nn.Linear(512, D)
        self.sim = nn.Linear(1, D)
        self.pos = nn.Parameter(torch.zeros(1, (SEARCH_IN // 16) ** 2, D))
        self.frame = nn.Embedding(2, D)
        self.kind = nn.Embedding(5, D)  # 0 search, 1 template, 2 candidate, 3 neighbour, 4 state
        self.cand, self.nb, self.st = MLP(6), MLP(6), MLP(8)
        self.query = nn.Parameter(torch.zeros(1, 1, D))
        layer = nn.TransformerEncoderLayer(D, 4, 256, 0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.exist, self.box, self.ptr_q, self.ptr_reg = nn.Linear(D, 1), nn.Linear(D, 4), nn.Linear(D, D), nn.Linear(D, 1)
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.query, std=0.02)

    def load_body(self, seed_state):
        self.body.load_state_dict({k[5:]: v for k, v in seed_state.items() if k.startswith("body.")})

    def forward(self, srch, tmpl, cand, cmask, nb, nmask, state):
        B = srch.shape[0]
        tv = self.body(tmpl)
        tv = self.tproj(torch.cat((tv.mean((2, 3)), tv.amax((2, 3))), 1))  # B x D
        frames = [0, 1] if self.use_prev else [0]
        toks, masks = [self.query.expand(B, -1, -1)], [torch.zeros(B, 1, dtype=torch.bool, device=srch.device)]
        for k in frames:
            f = self.proj(self.body(srch[:, k])).flatten(2).transpose(1, 2)  # B x 64 x D
            s = F.cosine_similarity(f, tv[:, None], dim=-1)[..., None]
            toks.append(f + self.pos + self.frame.weight[k] + self.sim(s) + self.kind.weight[0])
            masks.append(torch.zeros(B, f.shape[1], dtype=torch.bool, device=srch.device))
        toks.append((tv + self.kind.weight[1])[:, None])
        masks.append(torch.zeros(B, 1, dtype=torch.bool, device=srch.device))
        n_c = 0
        if self.use_cand:
            toks.append(self.cand(cand) + self.kind.weight[2]); masks.append(~cmask); n_c = cand.shape[1]
        if self.use_nb:
            toks.append(self.nb(nb) + self.kind.weight[3]); masks.append(~nmask)
        toks.append((self.st(state) + self.kind.weight[4])[:, None])
        masks.append(torch.zeros(B, 1, dtype=torch.bool, device=srch.device))
        x = self.enc(torch.cat(toks, 1), src_key_padding_mask=torch.cat(masks, 1))
        q = x[:, 0]
        out = {"exist": self.exist(q)[:, 0], "box": self.box(q)}
        reg = self.ptr_reg(q)  # B x 1
        if self.use_cand:
            start = 1 + 64 * len(frames) + 1
            c = x[:, start:start + n_c]
            lg = (self.ptr_q(q)[:, None] * c).sum(-1) / D ** 0.5
            lg = lg.masked_fill(~cmask, -1e4)
            out["ptr"] = torch.cat((lg, reg), 1)  # last index = use the regressed box
        else:
            out["ptr"] = reg
        return out


def batch_tensors(samples, device):
    st = lambda k: torch.as_tensor(np.stack([s[k] for s in samples])).to(device)
    return (to_tensor(np.stack([s["srch"] for s in samples])).to(device), to_tensor(np.stack([s["tmpl"] for s in samples])).to(device),
            st("cand"), st("cmask"), st("nb"), st("nmask"), st("state"))


def decode_box(box, side, centre):
    """Regressed box (dx, dy, log w, log h relative to side) -> tlwh."""
    w, h = side * np.exp(box[:, 2]), side * np.exp(box[:, 3])
    cx, cy = centre[:, 0] + box[:, 0] * side, centre[:, 1] + box[:, 1] * side
    return np.column_stack((cx - w / 2, cy - h / 2, w, h))


def load(path, device):
    ck = torch.load(path, map_location=device)
    m = STRN(**ck["flags"]).to(device)
    m.load_state_dict(ck["model"])
    return m.eval()


@torch.no_grad()
def predict(model, samples, device):
    """samples from inputs() -> (p_exist, boxes tlwh to write)."""
    out = model(*batch_tensors(samples, device))
    p = torch.sigmoid(out["exist"]).cpu().numpy().astype(float)
    side = np.array([s["side"] for s in samples], float)
    centre = np.stack([s["centre"] for s in samples]).astype(float)
    reg = decode_box(out["box"].cpu().numpy().astype(float), side, centre)
    pick = out["ptr"].argmax(1).cpu().numpy()
    boxes = reg.copy()
    if model.use_cand:
        for i, s in enumerate(samples):
            if pick[i] < N_CAND and s["cmask"][pick[i]]:
                boxes[i] = s["cbox"][pick[i]]
    return p, boxes
