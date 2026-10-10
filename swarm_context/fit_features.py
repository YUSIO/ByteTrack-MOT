"""Fit small models on explicit box features and compare feature sets (Exp066).

Each model is logit(detector score) + MLP(features); the feature sets are the box's own-chain features (own) and
own plus swarm-state features (own+swarm), plus own plus one group of swarm features at a time. Several seeds are
averaged. Reports the separation of targets among boxes scored below 0.7 on the validation sequences and a bootstrap
over sequences for the differences. With --save the fitted models of the named feature sets are stored.
"""
import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from swarm_context.data import Sequence
from swarm_context.features import GROUPS, OWN, SWARM, extract
from swarm_context.train import auc, average_precision, matched_fp


def load(job):
    s, min_score, cap, k = job
    seq = Sequence(s["name"], s["det"], s["seq_dir"], min_score, cap, tuple(s["frames"]) if s.get("frames") else None)
    return s["name"], s["role"], extract(seq, k)


def fit(xtr, ytr, wtr, btr, seed, epochs=30, hidden=64, drop=0.3, lr=2e-3, wd=1e-3, device="cpu"):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(xtr.shape[1], hidden), torch.nn.GELU(), torch.nn.Dropout(drop), torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Dropout(drop), torch.nn.Linear(hidden, 1)).to(device)
    torch.nn.init.zeros_(net[-1].weight)
    torch.nn.init.zeros_(net[-1].bias)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    x, y, w, b = (torch.as_tensor(v, dtype=torch.float32, device=device) for v in (xtr, ytr, wtr, btr))
    n = len(x)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (n // 4096 + 1))
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(b[idx] + net(x[idx]).squeeze(-1), y[idx], reduction="none") * w[idx]).sum() / w[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return net.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--min-score", type=float, default=0.01)
    ap.add_argument("--max-per-frame", type=int, default=48)
    ap.add_argument("--band-weight", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--save", nargs="*", default=[])
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(a.spec.read_text())
    with ProcessPoolExecutor(a.workers) as ex:
        data = list(ex.map(load, [(s, a.min_score, a.max_per_frame, a.k) for s in spec]))
    cols = OWN + SWARM

    def stack(role):
        parts = [d for _, r, d in data if r == role]
        x = np.concatenate([np.concatenate([d["own"], d["swarm"]], 1) for d in parts])
        return x, np.concatenate([d["lab"] for d in parts]), np.concatenate([d["score"] for d in parts]), np.concatenate([np.full(len(d["lab"]), i) for i, d in enumerate(parts)])

    xtr, ytr, str_, _ = stack("train")
    xva, yva, sva, qva = stack("val")
    keep = ytr >= 0
    xtr, ytr, str_ = xtr[keep], ytr[keep], str_[keep]
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    lg = lambda s: np.log(np.clip(s, 1e-4, 1 - 1e-4) / (1 - np.clip(s, 1e-4, 1 - 1e-4)))
    sets = {"own": OWN, "own+swarm": cols} | {f"own+{g}": OWN + f for g, f in GROUPS.items()} | {"swarm_only": ["logit"] + SWARM}
    pred, saved = {"detector": sva}, {}
    for name, feats in sets.items():
        idx = [cols.index(f) for f in feats]
        ps = []
        for seed in range(a.seeds):
            net = fit(((xtr - mu) / sd)[:, idx], ytr, np.where(str_ < 0.7, a.band_weight, 1.0), lg(str_), seed)
            with torch.no_grad():
                ps.append(torch.sigmoid(torch.as_tensor(lg(sva), dtype=torch.float32) + net(torch.as_tensor(((xva - mu) / sd)[:, idx], dtype=torch.float32)).squeeze(-1)).numpy())
            if name in a.save:
                saved.setdefault(name, []).append({k_: v.cpu() for k_, v in net.state_dict().items()})
        pred[name] = np.mean(ps, 0)
    if saved:
        with open(a.out / "models.pkl", "wb") as fh:
            pickle.dump({"columns": cols, "sets": {n: sets[n] for n in saved}, "mu": mu, "sd": sd, "states": saved, "k": a.k, "min_score": a.min_score, "max_per_frame": a.max_per_frame}, fh)

    def metric(p, sel):
        y, s, v = yva[sel], sva[sel], p[sel]
        m = y >= 0
        y, s, v = y[m], s[m], v[m]
        band = s < 0.7
        return {"ap_band": average_precision(v[band], y[band]), "auc_band": auc(v[band], y[band]), "true_at_fp_of_0.6": matched_fp(v, s, y, 0.6)["true_new"],
                "true_at_fp_of_0.3": matched_fp(v, s, y, 0.3)["true_new"], "true_at_fp_of_0.7": matched_fp(v, s, y, 0.7)["true_new"]}

    every = np.ones(len(yva), bool)
    res = {"train_boxes": int(len(ytr)), "val_boxes": int((yva >= 0).sum()), "val_band_targets": int(((yva == 1) & (sva < 0.7)).sum()), "scorings": {n: metric(p, every) for n, p in pred.items()}, "differences": {}}
    rng = np.random.default_rng(66)
    nseq = int(qva.max()) + 1
    masks = [qva == i for i in range(nseq)]
    for x, y in [("own", "detector"), ("own+swarm", "detector"), ("own+swarm", "own")] + [(f"own+{g}", "own") for g in GROUPS]:
        d = {"ap_band": [], "true_at_fp_of_0.6": []}
        for _ in range(a.boot):
            pick = rng.integers(0, nseq, nseq)
            sel = np.concatenate([np.flatnonzero(masks[i]) for i in pick])
            mx, my = metric(pred[x], sel), metric(pred[y], sel)
            for k_ in d:
                d[k_].append(mx[k_] - my[k_])
        res["differences"][f"{x} - {y}"] = {k_: {"mean": float(np.mean(v)), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for k_, v in d.items()}
    (a.out / "features_fit.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k_: v for k_, v in res.items() if k_ not in ("scorings", "differences")}))
    for n, m in res["scorings"].items():
        print(f"{n:18s}", {k_: round(v, 4) if isinstance(v, float) else v for k_, v in m.items()})
    for n, m in res["differences"].items():
        print(f"{n:28s}", {k_: (round(v["mean"], 4), [round(x, 4) for x in v["ci95"]]) for k_, v in m.items()})


if __name__ == "__main__":
    main()
