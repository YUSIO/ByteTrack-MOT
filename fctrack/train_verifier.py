"""Train a verification network (paper: ResNet18 on 224 x 128 crops; seed: scale-normalised crop with box regression).

Reads annotated training frames only. Sequences named in `held_out` are used for validation, never for fitting.
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
    ps, ys, ious = [], [], []
    for batch in loader:
        batch = [b.to(device) for b in batch]
        if kind == "paper":
            x, y = batch
            p = torch.sigmoid(model(x))
        else:
            x, sc, y, t = batch
            out = model(x, sc)
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
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--kind", choices=("paper", "seed"), required=True)
    ap.add_argument("--data", required=True, help="key of the training set in the config")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
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
    train_set = FrameSamples(fit, a.kind, tr["per_frame"], True, step=tr["frame_step"], preload=pre)
    val_set = FrameSamples(val, a.kind, tr["per_frame"], False, step=tr["val_frame_step"], seed=tr["seed"], preload=pre) if val else None
    info = {"kind": a.kind, "data": a.data, "fit_sequences": [d.name for d in fit], "held_out_sequences": [d.name for d in val],
            "fit_frames": len(train_set), "held_out_frames": len(val_set) if val_set else 0, "device": a.device, "torch": torch.__version__}
    print(json.dumps(info), flush=True)
    loader = DataLoader(train_set, batch_size=tr["frames_per_batch"], shuffle=True, num_workers=tr["workers"], collate_fn=collate, drop_last=True, persistent_workers=True)
    vloader = DataLoader(val_set, batch_size=tr["frames_per_batch"], shuffle=False, num_workers=tr["workers"], collate_fn=collate) if val_set else None
    model = (PaperVerifier(pretrained=tr.get("pretrained", True)) if a.kind == "paper" else SeedVerifier()).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, tr["lr"], total_steps=tr["epochs"] * len(loader), pct_start=0.1)
    log, t0 = [], time.time()
    for ep in range(tr["epochs"]):
        model.train()
        tot, steps = 0.0, 0
        for batch in loader:
            batch = [b.to(a.device, non_blocking=True) for b in batch]
            if a.kind == "paper":
                x, y = batch
                loss = F.binary_cross_entropy_with_logits(model(x), y)
            else:
                x, sc, y, t = batch
                out = model(x, sc)
                use, pos = y >= 0, y > 0.5
                loss = F.binary_cross_entropy_with_logits(out[use, 0], y[use])
                if pos.any():
                    loss = loss + 2.0 * F.smooth_l1_loss(out[pos, 1:], t[pos], beta=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot, steps = tot + loss.item(), steps + 1
        rec = {"epoch": ep + 1, "train_loss": tot / steps, "seconds": int(time.time() - t0)}
        if vloader is not None:
            rec.update(validate(a.kind, model, vloader, a.device))
        log.append(rec)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
        torch.save(model.state_dict(), a.output / "last.pt")
    (a.output / "train_log.json").write_text(json.dumps({"info": info, "config": tr, "epochs": log}, indent=1) + "\n")


if __name__ == "__main__":
    main()
