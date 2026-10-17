"""Exp087: train the STRN second version on logged real lost-track states (npz files of fctrack.strn2_data).

Loss = BCE(existence) + on positives CE(pointer over the 17 options). AdamW (weight decay 1e-4), visual body lr 1e-4 and
the rest 3e-4, 300 warm-up steps then cosine to 0, `epochs` epochs, batch 64, seed 0; augmentation: horizontal flip
around the proposal centre (p = 0.5) and per-sample brightness / contrast jitter. The last epoch is kept; no held-out
metric is used for selection. TensorBoard scalars (loss terms every 20 steps, learning rates, epoch means, data counts)
are written to <output>/tb.
Usage: python -m fctrack.train_strn2 --data a.npz b.npz --seed-weights seed.pt --output dir [--no-prev] [--no-nb]
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from fctrack.strn import N_CAND, flip, to_tensor
from fctrack.strn2 import STRN2

KEYS = ("srch", "tmpl", "cand", "cmask", "cbox", "nb", "nmask", "state", "exist", "ptr_t")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", type=Path, required=True)
    ap.add_argument("--seed-weights", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-prev", action="store_true")
    ap.add_argument("--no-nb", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    parts = [np.load(p) for p in a.data]
    data = {k: np.concatenate([p[k] for p in parts if p["exist"].size]) for k in KEYS}
    n = len(data["exist"])
    flags = {"use_prev": not a.no_prev, "use_nb": not a.no_nb}
    dev = torch.device(a.device)
    model = STRN2(**flags).to(dev)
    model.load_body(torch.load(a.seed_weights, map_location="cpu"))
    body = [p for n_, p in model.named_parameters() if n_.startswith("body.")]
    rest = [p for n_, p in model.named_parameters() if not n_.startswith("body.")]
    opt = torch.optim.AdamW([{"params": body, "lr": 1e-4}, {"params": rest, "lr": 3e-4}], weight_decay=1e-4)
    base = [g["lr"] for g in opt.param_groups]
    steps = a.epochs * math.ceil(n / a.batch)
    a.output.mkdir(parents=True, exist_ok=False)
    tb = SummaryWriter(str(a.output / "tb"))
    tb.add_text("run", json.dumps({"flags": flags, "samples": n, "positives": int(data["exist"].sum()), "data": [str(p) for p in a.data]}))
    tb.add_scalar("data/samples", n, 0); tb.add_scalar("data/positives", int(data["exist"].sum()), 0)
    log, step, t0 = [], 0, time.time()
    for ep in range(a.epochs):
        model.train()
        order = rng.permutation(n)
        tot = {"loss": 0.0, "exist": 0.0, "ptr": 0.0, "box": 0.0, "n": 0}
        for i in range(0, n, a.batch):
            idx = order[i:i + a.batch]
            samples = [{k: data[k][j] for k in KEYS} for j in idx]
            samples = [flip(s) if rng.random() < 0.5 else s for s in samples]
            st = lambda k: torch.as_tensor(np.stack([s[k] for s in samples])).to(dev)
            srch, tmpl = to_tensor(np.stack([s["srch"] for s in samples])).to(dev), to_tensor(np.stack([s["tmpl"] for s in samples])).to(dev)
            B = len(samples)
            ga = torch.as_tensor(rng.uniform(0.8, 1.2, B), dtype=torch.float32, device=dev)
            gb = torch.as_tensor(rng.uniform(-0.1, 0.1, B), dtype=torch.float32, device=dev)
            srch = (srch * ga[:, None, None, None, None] + gb[:, None, None, None, None]).clamp(0, 1)
            tmpl = (tmpl * ga[:, None, None, None] + gb[:, None, None, None]).clamp(0, 1)
            out = model(srch, tmpl, st("cand"), st("cmask"), st("nb"), st("nmask"), st("state"))
            y = st("exist")
            le = F.binary_cross_entropy_with_logits(out["exist"], y)
            pos = y > 0.5
            lp = lb = torch.zeros((), device=dev)
            if pos.any():
                lp = F.cross_entropy(out["ptr"][pos], st("ptr_t")[pos])
            loss = le + lp + lb
            lr_scale = min(1.0, (step + 1) / 300) * 0.5 * (1 + math.cos(math.pi * step / steps))
            for g, b0 in zip(opt.param_groups, base):
                g["lr"] = b0 * lr_scale
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 20 == 0:
                tb.add_scalar("step/loss", float(loss.detach()), step); tb.add_scalar("step/exist", float(le.detach()), step)
                tb.add_scalar("step/pointer", float(lp.detach()), step); tb.add_scalar("step/box", float(lb.detach()), step)
                tb.add_scalar("lr/body", opt.param_groups[0]["lr"], step); tb.add_scalar("lr/rest", opt.param_groups[1]["lr"], step)
            step += 1
            tot["loss"] += float(loss.detach()) * B; tot["exist"] += float(le.detach()) * B; tot["ptr"] += float(lp.detach()) * B; tot["box"] += float(lb.detach()) * B; tot["n"] += B
        rec = {"epoch": ep + 1, **{k: tot[k] / tot["n"] for k in ("loss", "exist", "ptr", "box")}, "seconds": round(time.time() - t0, 1)}
        log.append(rec)
        for k in ("loss", "exist", "ptr", "box"):
            tb.add_scalar(f"epoch/{k}", rec[k], ep + 1)
        tb.flush()
        print(json.dumps(rec), flush=True)
    torch.save({"model": model.state_dict(), "flags": flags,
                "meta": {"samples": n, "positives": int(data["exist"].sum()), "epochs": a.epochs, "batch": a.batch,
                         "data": [str(p) for p in a.data], "pointer_to_detector_box": int(((data["exist"] > 0.5) & (data["ptr_t"] < N_CAND)).sum())}},
               a.output / "last.pt")
    (a.output / "train_log.json").write_text(json.dumps(log, indent=1) + "\n")
    tb.close()


if __name__ == "__main__":
    main()
