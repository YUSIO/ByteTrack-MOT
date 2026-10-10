"""Rescore a detection cache with a trained swarm-context model (Exp066). Causal: frame t uses frames t-k .. t only.

Writes <out>/<name>/det.txt with the same boxes and the new scores. Boxes below the model's minimum score, or beyond
its per-frame cap, are not written (they are below anything a tracker reads). Annotations are not read.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from swarm_context.data import Sequence, collate
from swarm_context.model import SwarmContext


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--roles", nargs="*", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--scope", default=None, help="restrict the attention at inference (ablation)")
    ap.add_argument("--keep-above", type=float, default=None, help="boxes the detector scored at or above this keep their score")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = SwarmContext(**state["cfg"]).to(a.device)
    model.load_state_dict(state["model"])
    model.eval()
    if a.scope:
        model.scope = a.scope
    k = state["cfg"]["k"]
    summary = {}
    for s in json.loads(a.spec.read_text()):
        if a.roles and s.get("role") not in a.roles:
            continue
        seq = Sequence(s["name"], s["det"], s["seq_dir"], state["min_score"], state["max_per_frame"], tuple(s["frames"]) if s.get("frames") else None, labelled=False,
                       feat_file=s["feat"] if state["cfg"].get("app") else None)
        rows = []
        frames = list(range(seq.first, seq.last + 1))
        for i in range(0, len(frames), a.batch):
            chunk = frames[i:i + a.batch]
            b = collate([seq.window(f, k) for f in chunk])
            b = {key: v.to(a.device) for key, v in b.items()}
            p = torch.sigmoid(model(b["box"], b["step"], b["wh"], b["valid"], b.get("feat")))
            for j, f in enumerate(chunk):
                sel = (b["valid"][j] & (b["step"][j] == 0)).cpu().numpy()
                box, new = b["box"][j].cpu().numpy()[sel], p[j].float().cpu().numpy()[sel]
                if a.keep_above is not None:
                    new = np.where(box[:, 4] >= a.keep_above, box[:, 4], new)
                rows += ["{},-1,{:.4f},{:.4f},{:.4f},{:.4f},{:.6f},-1,-1,-1\n".format(f, *bx[:4], sc) for bx, sc in zip(box, new)]
        (a.out / s["name"]).mkdir(parents=True, exist_ok=True)
        (a.out / s["name"] / "det.txt").write_text("".join(rows))
        summary[s["name"]] = len(rows)
    (a.out / "rescore_summary.json").write_text(json.dumps({"checkpoint": str(a.ckpt), "epoch": state["epoch"], "rows": summary}, indent=1) + "\n")
    print("rescored", len(summary), "sequences", sum(summary.values()), "rows")


if __name__ == "__main__":
    main()
