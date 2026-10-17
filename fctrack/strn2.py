"""STRN second version (Exp087). Differences from fctrack/strn.py (Exp086), everything else shared:
  - the verifier's box is a 17th candidate token (features: offset / size relative to the proposal, verifier
    probability in the score slot, IoU with tracked boxes, flag = 1); detector candidates carry flag = 0;
  - the state vector gains the verifier probability (9 values);
  - no box regression: the pointer chooses one of the 17 options and that box is written.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fctrack.strn import D, N_CAND, MLP, SEARCH_IN, _iou, batch_tensors, inputs
from fctrack.verifier import SeedVerifier

N_OPT = N_CAND + 1


def inputs2(s, img_t, img_p, img_r, det, tracked):
    """Exp086 inputs plus the verifier option. s must carry vbox (tlwh) and vp."""
    x = inputs(s, img_t, img_p, img_r, det, tracked)
    side, (cx, cy) = float(x["side"]), x["centre"]
    vb = np.asarray(s["vbox"], float)
    vc = vb[:2] + vb[2:] / 2
    vov = _iou(vb, tracked).max() if len(tracked) else 0.0
    cand = np.zeros((N_OPT, 7), np.float32)
    cand[:N_CAND, :6] = x["cand"]
    cand[N_CAND] = [(vc[0] - cx) / side, (vc[1] - cy) / side, np.log(max(vb[2], 1) / side), np.log(max(vb[3], 1) / side), s["vp"], vov, 1.0]
    cbox = np.zeros((N_OPT, 4), np.float32); cbox[:N_CAND] = x["cbox"]; cbox[N_CAND] = vb
    cmask = np.zeros(N_OPT, bool); cmask[:N_CAND] = x["cmask"]; cmask[N_CAND] = vb[2] > 1 and vb[3] > 1
    x.update(cand=cand, cbox=cbox, cmask=cmask, state=np.append(x["state"], np.float32(s["vp"])).astype(np.float32))
    return x


class STRN2(nn.Module):
    def __init__(self, use_prev=True, use_nb=True):
        super().__init__()
        self.use_prev, self.use_nb, self.use_cand = use_prev, use_nb, True
        self.body = SeedVerifier().body
        self.proj = nn.Conv2d(256, D, 1)
        self.tproj = nn.Linear(512, D)
        self.sim = nn.Linear(1, D)
        self.pos = nn.Parameter(torch.zeros(1, (SEARCH_IN // 16) ** 2, D))
        self.frame = nn.Embedding(2, D)
        self.kind = nn.Embedding(5, D)
        self.cand, self.nb, self.st = MLP(7), MLP(6), MLP(9)
        self.query = nn.Parameter(torch.zeros(1, 1, D))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, 4, 256, 0.1, batch_first=True), 2)
        self.exist, self.ptr_q = nn.Linear(D, 1), nn.Linear(D, D)
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.query, std=0.02)

    def load_body(self, seed_state):
        self.body.load_state_dict({k[5:]: v for k, v in seed_state.items() if k.startswith("body.")})

    def forward(self, srch, tmpl, cand, cmask, nb, nmask, state):
        B, dev = srch.shape[0], srch.device
        tv = self.body(tmpl)
        tv = self.tproj(torch.cat((tv.mean((2, 3)), tv.amax((2, 3))), 1))
        frames = [0, 1] if self.use_prev else [0]
        zeros = lambda n: torch.zeros(B, n, dtype=torch.bool, device=dev)
        toks, masks = [self.query.expand(B, -1, -1)], [zeros(1)]
        for k in frames:
            f = self.proj(self.body(srch[:, k])).flatten(2).transpose(1, 2)
            sim = F.cosine_similarity(f, tv[:, None], dim=-1)[..., None]
            toks.append(f + self.pos + self.frame.weight[k] + self.sim(sim) + self.kind.weight[0]); masks.append(zeros(f.shape[1]))
        toks.append((tv + self.kind.weight[1])[:, None]); masks.append(zeros(1))
        start = sum(t.shape[1] for t in toks)
        toks.append(self.cand(cand) + self.kind.weight[2]); masks.append(~cmask)
        if self.use_nb:
            toks.append(self.nb(nb) + self.kind.weight[3]); masks.append(~nmask)
        toks.append((self.st(state) + self.kind.weight[4])[:, None]); masks.append(zeros(1))
        x = self.enc(torch.cat(toks, 1), src_key_padding_mask=torch.cat(masks, 1))
        q, c = x[:, 0], x[:, start:start + cand.shape[1]]
        ptr = ((self.ptr_q(q)[:, None] * c).sum(-1) / D ** 0.5).masked_fill(~cmask, -1e4)
        return {"exist": self.exist(q)[:, 0], "ptr": ptr}


def load(path, device):
    ck = torch.load(path, map_location=device)
    m = STRN2(**ck["flags"]).to(device)
    m.load_state_dict(ck["model"])
    return m.eval()


@torch.no_grad()
def predict2(model, samples, device):
    """-> (p_exist, the chosen option's box tlwh)."""
    out = model(*batch_tensors(samples, device))
    p = torch.sigmoid(out["exist"]).cpu().numpy().astype(float)
    pick = out["ptr"].argmax(1).cpu().numpy()
    boxes = np.stack([s["cbox"][k] for s, k in zip(samples, pick)]).astype(float)
    return p, boxes
