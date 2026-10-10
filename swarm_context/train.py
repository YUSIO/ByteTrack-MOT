"""Train the swarm-context rescoring model on cached detections (Exp066).

spec: JSON list of {"name", "det", "seq_dir", "role": "train" | "val", optional "frames": [first, last]}.
Reports, on the validation sequences, how well the rescored boxes separate targets from non-targets, for all boxes and
for the boxes the detector scored below 0.7, against the detector's own score.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from swarm_context.data import Sequence, Windows, collate
from swarm_context.model import SwarmContext


def average_precision(score, y):
    order = np.argsort(-score, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / (np.arange(len(y)) + 1)
    return float((prec * y).sum() / max(1, y.sum()))


def auc(score, y):
    order = np.argsort(score, kind="stable")
    rank = np.empty(len(score))
    rank[order] = np.arange(1, len(score) + 1)
    pos = y.sum()
    return float((rank[y == 1].sum() - pos * (pos + 1) / 2) / max(1, pos * (len(y) - pos)))


def matched_fp(new, old, y, thr):
    """True boxes admitted by the new score at the threshold that admits as many false boxes as the old score does at thr."""
    fp_old, tp_old = int(((old >= thr) & (y == 0)).sum()), int(((old >= thr) & (y == 1)).sum())
    neg = np.sort(new[y == 0])[::-1]
    tau = neg[fp_old] if fp_old < len(neg) else -1.0
    return {"false": fp_old, "true_old": tp_old, "true_new": int(((new > tau) & (y == 1)).sum())}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    new, old, lab = [], [], []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        out = torch.sigmoid(model(b["box"], b["step"], b["wh"], b["valid"], b.get("feat")))
        sel = b["valid"] & (b["step"] == 0)
        new.append(out[sel].float().cpu().numpy())
        old.append(b["box"][..., 4][sel].cpu().numpy())
        lab.append(b["lab"][sel].cpu().numpy())
    new, old, lab = np.concatenate(new), np.concatenate(old), np.concatenate(lab)
    keep = lab >= 0
    new, old, lab = new[keep], old[keep], lab[keep]
    band = old < 0.7
    res = {"boxes": int(len(lab)), "targets": int(lab.sum()), "band_boxes": int(band.sum()), "band_targets": int(lab[band].sum()),
           "ap_all_old": average_precision(old, lab), "ap_all_new": average_precision(new, lab),
           "ap_band_old": average_precision(old[band], lab[band]), "ap_band_new": average_precision(new[band], lab[band]),
           "auc_band_old": auc(old[band], lab[band]), "auc_band_new": auc(new[band], lab[band]),
           "at_0.6": matched_fp(new, old, lab, 0.6), "at_0.7": matched_fp(new, old, lab, 0.7), "at_0.3": matched_fp(new, old, lab, 0.3)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scope", default="all", choices=["all", "tube", "peers", "self"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--key-drop", type=float, default=0.15)
    ap.add_argument("--node-abs", type=int, default=0)
    ap.add_argument("--peer-dir", type=int, default=0)
    ap.add_argument("--epoch-size", type=int, default=6000, help="windows per epoch, sampled so that every sequence counts the same")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--band-weight", type=float, default=3.0)
    ap.add_argument("--max-per-frame", type=int, default=48)
    ap.add_argument("--min-score", type=float, default=0.01)
    ap.add_argument("--app-mode", default="none", choices=["none", "peer", "hist", "peer+hist", "other"], help="how the detector's box features are used (needs feat in the spec)")
    ap.add_argument("--eval-every", type=int, default=1, help="validate every this many epochs (always after the last)")
    ap.add_argument("--r0", type=float, default=2.0)
    ap.add_argument("--r1", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=False)
    spec = json.loads(a.spec.read_text())
    seqs = {"train": [], "val": []}
    for s in spec:
        seqs[s["role"]].append(Sequence(s["name"], s["det"], s["seq_dir"], a.min_score, a.max_per_frame, tuple(s["frames"]) if s.get("frames") else None,
                                        feat_file=s["feat"] if a.app_mode != "none" else None))
    train_set = Windows(seqs["train"], a.k, True, a.seed)
    per_seq = np.bincount([i for i, _ in train_set.index])
    sampler = torch.utils.data.WeightedRandomSampler([1.0 / per_seq[i] for i, _ in train_set.index], num_samples=a.epoch_size, replacement=True)
    tr = torch.utils.data.DataLoader(train_set, batch_size=a.batch, sampler=sampler, collate_fn=collate, num_workers=4, drop_last=True)
    # the "other" control borrows the confident boxes of the next sample of the batch, so its validation batches are shuffled too
    va = torch.utils.data.DataLoader(Windows(seqs["val"], a.k, False), batch_size=a.batch, shuffle=a.app_mode == "other", collate_fn=collate, num_workers=4,
                                     generator=torch.Generator().manual_seed(0))
    app = 0
    if a.app_mode != "none":
        allf = np.concatenate([v for q in seqs["train"] for v in q.feat.values() if len(v)])
        app = allf.shape[1]
    model = SwarmContext(k=a.k, d=a.d, heads=a.heads, layers=a.layers, drop=a.drop, scope=a.scope, r0=a.r0, r1=a.r1, node_abs=bool(a.node_abs), peer_dir=bool(a.peer_dir), key_drop=a.key_drop,
                         app=app, app_mode=a.app_mode if app else "peer+hist")
    if app:
        model.app_mu.copy_(torch.from_numpy(allf.mean(0)))
        model.app_sd.copy_(torch.from_numpy(allf.std(0) + 1e-6))
    model = model.to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(tr), pct_start=0.1)
    (a.out / "args.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, indent=1) + "\n")
    log, best, best_tp = open(a.out / "log.jsonl", "w"), -1.0, -1
    base = evaluate(model, va, a.device)
    log.write(json.dumps({"epoch": 0, "val": base}) + "\n")
    print("epoch 0", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in base.items()}), flush=True)
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for b in tr:
            b = {k: v.to(a.device) for k, v in b.items()}
            out = model(b["box"], b["step"], b["wh"], b["valid"], b.get("feat"))
            m = b["valid"] & (b["lab"] >= 0)
            w = torch.where(b["box"][..., 4] < 0.7, a.band_weight, 1.0)[m]
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(out[m], b["lab"][m].float(), reduction="none") * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, n = tot + float(loss), n + 1
        if epoch % a.eval_every and epoch != a.epochs:
            print(f"epoch {epoch} loss {tot / n:.4f} ({round(time.time() - t0, 1)}s)", flush=True)
            continue
        val = evaluate(model, va, a.device)
        rec = {"epoch": epoch, "loss": tot / n, "seconds": round(time.time() - t0, 1), "val": val}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"epoch {epoch} loss {tot / n:.4f} band AP {val['ap_band_old']:.4f} -> {val['ap_band_new']:.4f} AUC {val['auc_band_old']:.4f} -> {val['auc_band_new']:.4f} "
              f"true@fp(0.6) {val['at_0.6']['true_old']} -> {val['at_0.6']['true_new']} all AP {val['ap_all_old']:.4f} -> {val['ap_all_new']:.4f} ({rec['seconds']}s)", flush=True)
        state = {"model": model.state_dict(), "cfg": model.cfg, "epoch": epoch, "val": val, "min_score": a.min_score, "max_per_frame": a.max_per_frame}
        torch.save(state, a.out / "last.pt")
        if val["ap_band_new"] > best:
            best = val["ap_band_new"]
            torch.save(state, a.out / "best.pt")
        if val["at_0.6"]["true_new"] > best_tp:
            best_tp = val["at_0.6"]["true_new"]
            torch.save(state, a.out / "best_tp.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
