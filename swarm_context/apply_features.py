"""Rescore a detection cache with the feature models stored by fit_features --save (Exp066). Causal, no annotations.

Writes <out>/<set>/<name>/det.txt for every stored feature set: the same boxes with the new scores (mean over seeds).
"""
import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from swarm_context.data import Sequence
from swarm_context.features import extract


def load(job):
    s, min_score, cap, k = job
    seq = Sequence(s["name"], s["det"], s["seq_dir"], min_score, cap, tuple(s["frames"]) if s.get("frames") else None, labelled=False)
    return s["name"], extract(seq, k)


def net(n_in, state):
    hidden = state["0.weight"].shape[0]
    m = torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.GELU(), torch.nn.Dropout(0.0), torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Dropout(0.0), torch.nn.Linear(hidden, 1))
    m.load_state_dict(state)
    return m.eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--roles", nargs="*", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    with open(a.models, "rb") as fh:
        m = pickle.load(fh)
    spec = [s for s in json.loads(a.spec.read_text()) if not a.roles or s.get("role") in a.roles]
    with ProcessPoolExecutor(a.workers) as ex:
        data = list(ex.map(load, [(s, m["min_score"], m["max_per_frame"], m["k"]) for s in spec]))
    lg = lambda s: np.log(np.clip(s, 1e-4, 1 - 1e-4) / (1 - np.clip(s, 1e-4, 1 - 1e-4)))
    for name, feats in m["sets"].items():
        idx = [m["columns"].index(f) for f in feats]
        nets = [net(len(idx), st) for st in m["states"][name]]
        rows_total = {}
        for seq_name, d in data:
            x = torch.as_tensor(((np.concatenate([d["own"], d["swarm"]], 1) - m["mu"]) / m["sd"])[:, idx], dtype=torch.float32)
            base = torch.as_tensor(lg(d["score"]), dtype=torch.float32)
            new = np.mean([torch.sigmoid(base + n(x).squeeze(-1)).numpy() for n in nets], 0) if len(x) else np.zeros(0)
            rows = ["{},-1,{:.4f},{:.4f},{:.4f},{:.4f},{:.6f},-1,-1,-1\n".format(f, *bx, sc) for f, bx, sc in zip(d["frame"], d["box"], new)]
            (a.out / name.replace("+", "_") / seq_name).mkdir(parents=True, exist_ok=True)
            (a.out / name.replace("+", "_") / seq_name / "det.txt").write_text("".join(rows))
            rows_total[seq_name] = len(rows)
        (a.out / name.replace("+", "_") / "rescore_summary.json").write_text(json.dumps({"models": str(a.models), "set": name, "seeds": len(nets), "rows": rows_total}, indent=1) + "\n")
        print("rescored", name, sum(rows_total.values()), "rows")


if __name__ == "__main__":
    main()
