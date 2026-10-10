"""Open-loop comparison of rescoring models on labelled sequences (Exp066).

For every box of the listed sequences (causal windows, as in apply.py) it collects the detector's score, each model's
score and the label, then reports the separation of targets from non-targets among the boxes the detector scored
below 0.7, with a bootstrap over sequences for the difference between any two scorings.
--override name=scope evaluates a checkpoint with its attention restricted at inference only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from swarm_context.data import Sequence, collate
from swarm_context.model import SwarmContext
from swarm_context.train import auc, average_precision, matched_fp


@torch.no_grad()
def scores(ckpt, seqs, device, scope=None, batch=64):
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = SwarmContext(**state["cfg"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    if scope:
        model.scope = scope
    out = []
    for seq in seqs:
        frames, res = list(range(seq.first, seq.last + 1)), []
        for i in range(0, len(frames), batch):
            b = collate([seq.window(f, state["cfg"]["k"]) for f in frames[i:i + batch]])
            b = {k: v.to(device) for k, v in b.items()}
            p = torch.sigmoid(model(b["box"], b["step"], b["wh"], b["valid"], b.get("feat")))
            res.append(p[b["valid"] & (b["step"] == 0)].float().cpu().numpy())
        out.append(np.concatenate(res))
    return out, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--roles", nargs="*", default=["val"])
    ap.add_argument("--model", action="append", required=True, help="name=checkpoint")
    ap.add_argument("--override", action="append", default=[], help="name=scope, evaluate that model with this attention scope")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    models = dict(m.split("=", 1) for m in a.model)
    over = dict(o.split("=", 1) for o in a.override)
    spec = [s for s in json.loads(a.spec.read_text()) if s.get("role") in a.roles]
    first = torch.load(next(iter(models.values())), map_location="cpu", weights_only=False)
    seqs = [Sequence(s["name"], s["det"], s["seq_dir"], first["min_score"], first["max_per_frame"], tuple(s["frames"]) if s.get("frames") else None, feat_file=s.get("feat") if first["cfg"].get("app") else None) for s in spec]
    old = [np.concatenate([q.det[f][:, 4] for f in range(q.first, q.last + 1)]) for q in seqs]
    lab = [np.concatenate([q.lab[f] for f in range(q.first, q.last + 1)]) for q in seqs]
    S = {"detector": old}
    for name, ck in models.items():
        S[name], _ = scores(ck, seqs, a.device)
        if name in over:
            S[f"{name}@{over[name]}"], _ = scores(ck, seqs, a.device, scope=over[name])

    def metric(sc, idx):
        s, o, y = (np.concatenate([x[i] for i in idx]) for x in (sc, old, lab))
        k = y >= 0
        s, o, y = s[k], o[k], y[k]
        band = o < 0.7
        return {"ap_band": average_precision(s[band], y[band]), "auc_band": auc(s[band], y[band]), "ap_all": average_precision(s, y),
                "true_at_fp_of_0.6": matched_fp(s, o, y, 0.6)["true_new"], "true_at_fp_of_0.7": matched_fp(s, o, y, 0.7)["true_new"], "true_at_fp_of_0.3": matched_fp(s, o, y, 0.3)["true_new"]}

    every = list(range(len(seqs)))
    res = {"sequences": [q.name for q in seqs], "boxes": int(sum(len(x) for x in old)), "scorings": {n: metric(sc, every) for n, sc in S.items()}, "differences": {}}
    rng = np.random.default_rng(66)
    draws = [rng.integers(0, len(seqs), len(seqs)) for _ in range(a.boot)]
    names = list(S)
    for i, x in enumerate(names):
        for y in names[:i]:
            d = {k: [] for k in ("ap_band", "true_at_fp_of_0.6")}
            for idx in draws:
                mx, my = metric(S[x], idx), metric(S[y], idx)
                for k in d:
                    d[k].append(mx[k] - my[k])
            res["differences"][f"{x} - {y}"] = {k: {"mean": float(np.mean(v)), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for k, v in d.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    for n, m in res["scorings"].items():
        print(n, {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
    for n, m in res["differences"].items():
        print(n, {k: (round(v["mean"], 4), [round(x, 4) for x in v["ci95"]]) for k, v in m.items()})


if __name__ == "__main__":
    main()
