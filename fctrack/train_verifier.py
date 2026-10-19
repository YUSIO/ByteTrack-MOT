"""Train a verification network (paper: ResNet18 on 224 x 128 crops; paper_resized: the same on scale-normalised crops
resized to 224 x 128; seed: scale-normalised crop with box regression).

Reads annotated training frames only. Sequences named in `held_out` are used for validation, never for fitting.
Exp112: with `--birth-candidates` (detector boxes on the training frames, labelled by IoU with the annotation) the seed
network gets a sixth output trained only on those boxes (correct birth or not); TensorBoard logs go to <output>/tb.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fctrack.verifier import FrameSamples, PaperVerifier, SeedVerifier


def read_birth(path):
    """CSV seq,frame,x,y,w,h,score,label,gt_index -> {(seq, frame): n x 6 [x, y, w, h, label, gt_index]}"""
    import csv
    out = {}
    for r in csv.DictReader(open(path)):
        out.setdefault((r["seq"], int(r["frame"])), []).append([float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"]), float(r["label"]), float(r["gt_index"])])
    return {k: np.array(v, np.float32) for k, v in out.items()}


def collate(batch):
    return tuple(torch.cat(parts) for parts in zip(*batch))


def auc(score, label):
    order = np.argsort(score)
    rank = np.empty(len(score))
    rank[order] = np.arange(1, len(score) + 1)
    pos = label > 0.5
    return float((rank[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / max(1, pos.sum() * (~pos).sum()))


@torch.no_grad()
def validate(kind, model, loader, device):
    model.eval()
    ps, ys, ious, bps, bys, b0 = [], [], [], [], [], []
    for batch in loader:
        batch = [b.to(device) for b in batch]
        if kind.startswith("paper"):
            x, y = batch
            p = torch.sigmoid(model(x))
        else:
            x, sc, y, t, yb = batch
            out = model(x, sc)
            usb = yb >= 0
            if usb.any():  # birth candidates: the birth output, and the fill-in output on the same boxes for comparison
                if out.shape[1] > 5:
                    bps.append(torch.sigmoid(out[usb, 5]).cpu())
                b0.append(torch.sigmoid(out[usb, 0]).cpu())
                bys.append(yb[usb].cpu())
                y = torch.where(usb, torch.full_like(y, -1.0), y)  # the fill-in metrics stay on the original seeds
            p = torch.sigmoid(out[:, 0])
            keep = y > 0.5
            if keep.any():  # IoU of the regressed box with the target box, both in seed units (seed w = seed h = 1)
                o, g = out[keep, 1:], t[keep]
                pw, ph, gw, gh = torch.exp(o[:, 2]), torch.exp(o[:, 3]), torch.exp(g[:, 2]), torch.exp(g[:, 3])
                ix = (torch.minimum(o[:, 0] + pw / 2, g[:, 0] + gw / 2) - torch.maximum(o[:, 0] - pw / 2, g[:, 0] - gw / 2)).clamp(min=0)
                iy = (torch.minimum(o[:, 1] + ph / 2, g[:, 1] + gh / 2) - torch.maximum(o[:, 1] - ph / 2, g[:, 1] - gh / 2)).clamp(min=0)
                ious.append((ix * iy / (pw * ph + gw * gh - ix * iy)).cpu())
        use = y >= 0
        ps.append(p[use].cpu())
        ys.append(y[use].cpu())
    p, y = torch.cat(ps).numpy(), torch.cat(ys).numpy()
    res = {"samples": int(len(y)), "positives": int((y > 0.5).sum()), "auc": auc(p, y), "accuracy_at_0.5": float(((p >= 0.5) == (y > 0.5)).mean())}
    for thr in (0.5, 0.9, 0.97):
        res[f"recall_at_p{thr}"] = float((p[y > 0.5] >= thr).mean())
        res[f"false_accept_at_p{thr}"] = float((p[y < 0.5] >= thr).mean())
    if ious:
        iou = torch.cat(ious).numpy()
        res["regressed_iou_mean"], res["regressed_iou_ge_0.5"] = float(iou.mean()), float((iou >= 0.5).mean())
    if bys:
        yb = torch.cat(bys).numpy()
        res["birth_samples"], res["birth_positives"] = int(len(yb)), int((yb > 0.5).sum())
        res["birth_auc_fillin_output"] = auc(torch.cat(b0).numpy(), yb)
        if bps:
            pb = torch.cat(bps).numpy()
            res["birth_auc"] = auc(pb, yb)
            for thr in (0.5, 0.9):
                acc = pb >= thr
                res[f"birth_precision_at_{thr}"] = float(yb[acc].mean()) if acc.any() else float("nan")
                res[f"birth_recall_at_{thr}"] = float(acc[yb > 0.5].mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--kind", choices=("paper", "paper_resized", "seed"), required=True)
    ap.add_argument("--data", required=True, help="key of the training set in the config")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--birth-candidates", type=Path, default=None, help="Exp112: CSV of labelled detector boxes on the training frames")
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    tr, data = cfg["train"][a.kind], cfg["train_data"][a.data]
    cv2.setNumThreads(1)  # the loader workers must not each start an OpenCV thread pool
    torch.manual_seed(tr["seed"])
    np.random.seed(tr["seed"])
    root = Path(data["root"])
    seqs = sorted(d for d in root.iterdir() if (d / "gt/gt.txt").exists())
    held = set(data["held_out"])
    fit, val = [d for d in seqs if d.name not in held], [d for d in seqs if d.name in held]
    a.output.mkdir(parents=True, exist_ok=False)
    pre = tr.get("preload", False)
    birth = read_birth(a.birth_candidates) if a.birth_candidates else None
    bpf = tr.get("birth_per_frame", 0) if birth else 0
    train_set = FrameSamples(fit, a.kind, tr["per_frame"], True, step=tr["frame_step"], preload=pre, birth=birth, birth_per_frame=bpf)
    val_set = FrameSamples(val, a.kind, tr["per_frame"], False, step=tr["val_frame_step"], seed=tr["seed"], preload=pre, birth=birth, birth_per_frame=bpf) if val else None
    info = {"kind": a.kind, "data": a.data, "fit_sequences": [d.name for d in fit], "held_out_sequences": [d.name for d in val],
            "fit_frames": len(train_set), "held_out_frames": len(val_set) if val_set else 0, "device": a.device, "torch": torch.__version__}
    print(json.dumps(info), flush=True)
    loader = DataLoader(train_set, batch_size=tr["frames_per_batch"], shuffle=True, num_workers=tr["workers"], collate_fn=collate, drop_last=True, persistent_workers=True)
    vloader = DataLoader(val_set, batch_size=tr["frames_per_batch"], shuffle=False, num_workers=tr["workers"], collate_fn=collate) if val_set else None
    model = (PaperVerifier(pretrained=tr.get("pretrained", True)) if a.kind.startswith("paper") else SeedVerifier(birth=bool(bpf))).to(a.device)
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(str(a.output / "tb"))
    gstep = 0
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, tr["lr"], total_steps=tr["epochs"] * len(loader), pct_start=0.1)
    log, t0 = [], time.time()
    for ep in range(tr["epochs"]):
        model.train()
        tot, steps = 0.0, 0
        for batch in loader:
            batch = [b.to(a.device, non_blocking=True) for b in batch]
            if a.kind.startswith("paper"):
                x, y = batch
                loss = F.binary_cross_entropy_with_logits(model(x), y)
            else:
                x, sc, y, t, yb = batch
                out = model(x, sc)
                use, pos = y >= 0, y > 0.5
                loss = F.binary_cross_entropy_with_logits(out[use, 0], y[use])
                if pos.any():
                    loss = loss + 2.0 * F.smooth_l1_loss(out[pos, 1:5], t[pos], beta=0.05)
                usb = yb >= 0
                if out.shape[1] > 5 and usb.any():
                    lb = F.binary_cross_entropy_with_logits(out[usb, 5], yb[usb])
                    loss = loss + lb
                    tb.add_scalar("train/birth_bce", lb.item(), gstep)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot, steps = tot + loss.item(), steps + 1
            tb.add_scalar("train/loss", loss.item(), gstep)
            tb.add_scalar("train/lr", sched.get_last_lr()[0], gstep)
            gstep += 1
        rec = {"epoch": ep + 1, "train_loss": tot / steps, "seconds": int(time.time() - t0)}
        if vloader is not None:
            rec.update(validate(a.kind, model, vloader, a.device))
        log.append(rec)
        for k, v in rec.items():
            if k != "epoch" and isinstance(v, (int, float)) and np.isfinite(v):
                tb.add_scalar(("val/" if k not in ("train_loss", "seconds") else "epoch/") + k, v, ep + 1)
        tb.flush()
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
        torch.save(model.state_dict(), a.output / "last.pt")
    (a.output / "train_log.json").write_text(json.dumps({"info": info, "config": tr, "epochs": log}, indent=1) + "\n")


if __name__ == "__main__":
    main()
