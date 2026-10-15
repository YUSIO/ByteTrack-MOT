"""Train several track heads side by side on the features of a frozen detector (Exp071 stage 1).

Every batch goes through the detector once; each head sees the same windows and differs only in the input groups it
is given (jdt.head). Training reads annotated frames of the fit sequences that the detector did not train on;
validation samples (fixed random state) come from the held-out sequences. No detector or tracker output is read.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from fctrack.train_verifier import auc
from jdt.detector import features, load_detector, normalise
from jdt.head import TrackHead, decode_box, loss, window_features
from jdt.query import stack
from jdt.samples import TrackSamples, collate


def unseen_frames(val_json):
    """{sequence: frames} of the detector's validation json (frames the detector did not train on)."""
    d = json.loads(Path(val_json).read_text())
    out = {}
    for im in d["images"]:
        out.setdefault(im["file_name"].split("/")[0], []).append(int(im["frame_id"]))
    return out


def run_batch(det, heads, batch, device, amp=True):
    imgs, qs, qimg, B = batch
    if not qs:
        return None, None
    x = normalise(imgs.to(device, non_blocking=True))
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        p3, obj = features(det, x)
    q = stack(qs, device)
    qimg = qimg.to(device)
    fs, ob, tf = window_features(p3, obj, qimg, q["centre"], q["half"], p3, q["tbox"], qimg + B)
    return {n: h(fs, ob, tf, q) for n, h in heads.items()}, q


def iou_cxcywh(a, b):
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    iw = (torch.minimum(ax2, bx2) - torch.maximum(ax1, bx1)).clamp(min=0)
    ih = (torch.minimum(ay2, by2) - torch.maximum(ay1, by1)).clamp(min=0)
    inter = iw * ih
    return inter / (a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter)


@torch.no_grad()
def validate(det, heads, loader, device):
    for h in heads.values():
        h.eval()
    acc = {n: {"p": [], "y": [], "iou": [], "kind": []} for n in heads}
    for batch in loader:
        outs, q = run_batch(det, heads, batch, device)
        if outs is None:
            continue
        for n, o in outs.items():
            p, box = decode_box(o, q)
            acc[n]["p"].append(p.cpu())
            acc[n]["y"].append(q["exist"].cpu())
            acc[n]["kind"].append(q["kind"].cpu())
            pos = q["exist"] > 0.5
            acc[n]["iou"].append(iou_cxcywh(box[pos], q["gt"][pos]).cpu())
    res = {}
    for n, a in acc.items():
        p, y, iou, kind = torch.cat(a["p"]).numpy(), torch.cat(a["y"]).numpy(), torch.cat(a["iou"]).numpy(), torch.cat(a["kind"]).numpy()
        r = {"samples": int(len(y)), "positives": int((y > 0.5).sum()), "auc": auc(p, y), "box_iou_ge_0.5": float((iou >= 0.5).mean()), "box_iou_mean": float(iou.mean())}
        for thr in (0.5, 0.7, 0.9):
            r[f"recall_at_p{thr}"] = float((p[y > 0.5] >= thr).mean())
            r[f"false_accept_at_p{thr}"] = float((p[y < 0.5] >= thr).mean())
        for kd, nm in ((0, "jitter"), (1, "on_member"), (2, "displaced"), (3, "background")):
            sel = kind == kd
            r[f"n_{nm}"], r[f"pos_{nm}"] = int(sel.sum()), int((y[sel] > 0.5).sum())
            r[f"accept_p0.7_pos_{nm}"] = float((p[sel & (y > 0.5)] >= 0.7).mean()) if (sel & (y > 0.5)).any() else None
            r[f"accept_p0.7_neg_{nm}"] = float((p[sel & (y < 0.5)] >= 0.7).mean()) if (sel & (y < 0.5)).any() else None
        res[n] = r
    for h in heads.values():
        h.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: number of training items")
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    tr = cfg["train"]
    cv2.setNumThreads(1)
    torch.manual_seed(tr["seed"])
    np.random.seed(tr["seed"])
    root = Path(cfg["dataset_train_dir"])
    frames = unseen_frames(cfg["detector"]["val_json"])
    held = set(cfg["held_out"])
    seqs = sorted(d for d in root.iterdir() if (d / "gt/gt.txt").exists() and d.name in frames)
    fit, val = [d for d in seqs if d.name not in held], [d for d in seqs if d.name in held]
    train_set = TrackSamples(fit, frames, train=True)
    val_set = TrackSamples(val, frames, train=False, seed=tr["seed"])
    if a.limit:
        train_set.items = train_set.items[: a.limit]
        val_set.items = val_set.items[: max(8, a.limit // 4)]
    a.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(a.device)
    _, det = load_detector(cfg["detector"]["exp_file"], cfg["detector"]["ckpt"], device)
    for p in det.parameters():
        p.requires_grad_(False)
    heads = {n: TrackHead(g).to(device) for n, g in cfg["heads"].items()}
    info = {"fit_sequences": [d.name for d in fit], "held_out_sequences": [d.name for d in val], "fit_items": len(train_set), "held_out_items": len(val_set),
            "heads": cfg["heads"], "parameters_per_head": sum(p.numel() for p in next(iter(heads.values())).parameters()), "torch": torch.__version__}
    print(json.dumps(info), flush=True)
    loader = DataLoader(train_set, batch_size=tr["batch"], shuffle=True, num_workers=tr["workers"], collate_fn=collate, drop_last=True, persistent_workers=True)
    vloader = DataLoader(val_set, batch_size=tr["batch"], shuffle=False, num_workers=tr["workers"], collate_fn=collate)
    params = [p for h in heads.values() for p in h.parameters()]
    opt = torch.optim.AdamW(params, lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, tr["lr"], total_steps=tr["epochs"] * len(loader), pct_start=0.1)
    log, t0 = [], time.time()
    for ep in range(tr["epochs"]):
        tot = {n: 0.0 for n in heads}
        steps = 0
        for batch in loader:
            outs, q = run_batch(det, heads, batch, device)
            if outs is None:
                continue
            total = 0.0
            for n, o in outs.items():
                l, _ = loss(o, q)
                total = total + l
                tot[n] += l.item()
            opt.zero_grad()
            total.backward()
            opt.step()
            sched.step()
            steps += 1
        rec = {"epoch": ep + 1, "seconds": int(time.time() - t0), "train_loss": {n: v / max(steps, 1) for n, v in tot.items()}}
        rec["held_out"] = validate(det, heads, vloader, device)
        log.append(rec)
        print(json.dumps(rec), flush=True)
        torch.save({n: h.state_dict() for n, h in heads.items()}, a.output / "heads_last.pt")
    (a.output / "train_log.json").write_text(json.dumps({"info": info, "config": cfg, "epochs": log}, indent=1) + "\n")


if __name__ == "__main__":
    main()
