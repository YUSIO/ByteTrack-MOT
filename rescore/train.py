"""Exp095 (residual form of Exp094): train the temporal re-scoring network on a candidate table (build.py output), with sequence-level folds.

  train.py cv    --table dev.npz --folds 4 --out <dir>              # out-of-fold probabilities for every row
  train.py final --table dev.npz --apply test.npz --out <dir>       # one model on all rows, probabilities for another table
The label is "this box is the best-IoU box (IoU >= 0.5) of an annotated row"; the output is read as a calibrated probability.
Fold assignment: sorted unique sequence numbers, index mod folds. No validation-based epoch selection: the last epoch is kept.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

EPOCHS, LR, WD, BATCH, SEED = 12, 2e-3, 1e-4, 256, 94


def block(i, o, s):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Rescorer(nn.Module):
    def __init__(self, n_scalar):
        super().__init__()
        c = (32, 64, 96, 128)
        self.body = nn.Sequential(block(12, c[0], 1), block(c[0], c[1], 2), block(c[1], c[2], 2), block(c[2], c[3], 2))
        self.scal = nn.Sequential(nn.Linear(n_scalar, 64), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Linear(c[3] * 2 + 64, 128), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(128, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)  # starts as the detector score exactly

    def forward(self, x, s):
        f = self.body(x)
        delta = self.head(torch.cat((f.mean((2, 3)), f.amax((2, 3)), self.scal(s)), 1))[:, 0]
        return s[:, 0] + delta  # residual: logit of the detector score (scalar 0, see prep) plus a learned correction


def prep(t):
    s = t["scal"].copy()
    s[:, 0] = np.log(np.clip(s[:, 0], 1e-3, 1 - 1e-3) / (1 - np.clip(s[:, 0], 1e-3, 1 - 1e-3)))  # logit of the detector score
    s[:, 5:7] = np.log(np.clip(s[:, 5:7], 1e-3, 1 - 1e-3) / (1 - np.clip(s[:, 5:7], 1e-3, 1 - 1e-3))) * (s[:, 5:7] > 0)
    s[:, 7] = np.log1p(s[:, 7])
    return s.astype(np.float32)


def auc(score, y):
    from scipy.stats import rankdata
    r, p = rankdata(score), y > 0.5
    return float((r[p].sum() - p.sum() * (p.sum() + 1) / 2) / max(1, p.sum() * (~p).sum()))


def fit(x, s, y, device, tb=None):
    torch.manual_seed(SEED); np.random.seed(SEED)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(s), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True, num_workers=0)
    model = Rescorer(s.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * len(loader), pct_start=0.15)
    step = 0
    for ep in range(EPOCHS):
        model.train(); tot = 0.0
        for xb, sb, yb in loader:
            xb = xb.to(device).float() / 255.0
            if np.random.rand() < 0.5: xb = xb.flip(3)
            if np.random.rand() < 0.5: xb = xb.flip(2)
            xb = xb * float(np.random.uniform(0.85, 1.15)) + float(np.random.uniform(-0.08, 0.08))
            loss = F.binary_cross_entropy_with_logits(model(xb, sb.to(device)), yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item(); step += 1
            if tb: tb.add_scalar("loss/train_step", loss.item(), step)
        print(f"epoch {ep + 1} loss {tot / len(loader):.4f}", flush=True)
        if tb: tb.add_scalar("loss/train_epoch", tot / len(loader), ep + 1)
    return model


@torch.no_grad()
def predict(model, x, s, device):
    model.eval(); out = []
    for i in range(0, len(x), 1024):
        out.append(torch.sigmoid(model(torch.from_numpy(x[i:i + 1024]).to(device).float() / 255.0, torch.from_numpy(s[i:i + 1024]).to(device))).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("cv", "final"))
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--apply", type=Path)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    dev = torch.device(a.device)
    t = dict(np.load(a.table)); x, s, y, seq = t["crops"], prep(t), t["label"], t["seq"]
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(a.out / "tb"))
    except Exception:
        tb = None
    t0, log = time.time(), {}
    if a.mode == "cv":
        names = sorted(set(seq.tolist())); fold_of = {n: i % a.folds for i, n in enumerate(names)}
        fold = np.array([fold_of[n] for n in seq.tolist()]); p = np.zeros(len(y), np.float32)
        for k in range(a.folds):
            tr, te = fold != k, fold == k
            model = fit(x[tr], s[tr], y[tr], dev, tb)
            p[te] = predict(model, x[te], s[te], dev)
            log[f"fold{k}"] = {"train_rows": int(tr.sum()), "test_rows": int(te.sum()), "test_sequences": [n for n in names if fold_of[n] == k], "auc": auc(p[te], y[te]), "auc_detector_score": auc(t["scal"][te, 0], y[te])}
            print(k, log[f"fold{k}"], flush=True)
        np.savez(a.out / "oof.npz", p=p, seq=seq, frame=t["frame"], box=t["box"], score=t["scal"][:, 0], label=y, fold=fold)
    else:
        model = fit(x, s, y, dev, tb)
        torch.save(model.state_dict(), a.out / "model.pt")
        ap_ = dict(np.load(a.apply))
        p = predict(model, ap_["crops"], prep(ap_), dev)
        np.savez(a.out / "applied.npz", p=p, seq=ap_["seq"], frame=ap_["frame"], box=ap_["box"], score=ap_["scal"][:, 0], label=ap_["label"])
        log["applied_rows"] = int(len(p))
    log["seconds"] = int(time.time() - t0)
    (a.out / "train_log.json").write_text(json.dumps(log, indent=1) + "\n")


if __name__ == "__main__":
    main()
