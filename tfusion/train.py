"""Train the temporal fusion on a frozen YOLOX detector with the detector's own head loss (Exp091).

Run from the code root with PYTHONPATH=. :
  python tfusion/train.py -f <exp file> -c <detector ckpt> --view <view root>/train --sequences ... --gmc <dir> --align zero|gmc --output <dir>
Writes <output>/fusion_last.pt (and fusion_epoch<k>.pt), train_log.jsonl and TensorBoard events in <output>/tb.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tfusion.data import PairDataset
from tfusion.module import TemporalFusion
from yolox.exp import get_exp


def frozen_detector(exp_file, ckpt, device):
    exp = get_exp(exp_file, None)
    model = exp.get_model()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return exp, model


def head_loss_mode(head):
    """Head computes its training losses, but its BatchNorm layers keep the frozen running statistics."""
    head.train()
    for m in head.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--exp-file", required=True)
    ap.add_argument("-c", "--ckpt", required=True)
    ap.add_argument("--view", type=Path, required=True)
    ap.add_argument("--sequences", nargs="+", required=True)
    ap.add_argument("--gmc", type=Path, required=True)
    ap.add_argument("--align", choices=["zero", "gmc"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-iters", type=int, default=0, help="smoke test only")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    a.output.mkdir(parents=True, exist_ok=False)
    exp, model = frozen_detector(a.exp_file, a.ckpt, dev)
    size = tuple(exp.input_size)
    with torch.no_grad():
        chans = [f.shape[1] for f in model.backbone(torch.zeros(1, 3, *size, device=dev))]
    fusion = TemporalFusion(chans, a.align).to(dev)
    ds = PairDataset(a.view, a.sequences, a.gmc, size=size, seed=a.seed)
    opt = torch.optim.AdamW(fusion.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps = a.epochs * ((len(ds) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps, 1))
    tb = SummaryWriter(a.output / "tb")
    log = open(a.output / "train_log.jsonl", "w")
    meta = {"args": {k: str(v) for k, v in vars(a).items()}, "channels": chans, "pairs": len(ds), "steps": steps,
            "trainable_parameters": sum(p.numel() for p in fusion.parameters())}
    (a.output / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta), flush=True)
    step, started = 0, time.time()
    for epoch in range(a.epochs):
        ds.set_epoch(epoch)
        loader = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=False,
                                             generator=torch.Generator().manual_seed(a.seed * 1000 + epoch))
        fusion.train()
        sums, iters = {}, 0
        for x0, x1, m_inv, lab in loader:
            x0, x1, m_inv, lab = x0.to(dev), x1.to(dev), m_inv.to(dev), lab.to(dev)
            with torch.no_grad():
                f0, f1 = model.backbone(x0), model.backbone(x1)
            fused = fusion(list(f1), list(f0), m_inv, size)
            head_loss_mode(model.head)
            loss, iou_l, conf_l, cls_l, l1_l, nfg = model.head(fused, lab, x1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step(); step += 1; iters += 1
            vals = {"total": float(loss), "iou": float(iou_l), "conf": float(conf_l), "cls": float(cls_l), "num_fg": float(nfg)}
            for k, v in vals.items():
                sums[k] = sums.get(k, 0.0) + v
            if step % 20 == 0:
                for k, v in vals.items():
                    tb.add_scalar(f"train/{k}", v, step)
                tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
                for lvl, g in enumerate(fusion.gate_means(list(f1), list(f0), m_inv, size)):
                    tb.add_scalar(f"gate/mean_level{lvl}", g, step)
                tb.add_scalar("value/weight_norm", float(sum(v.weight.norm() for v in fusion.value)), step)
            if a.limit_iters and step >= a.limit_iters:
                break
        model.head.eval()
        n = max(1, iters)
        rec = {"epoch": epoch + 1, "step": step, "seconds": round(time.time() - started, 1), **{k: v / n for k, v in sums.items()}}
        for k, v in rec.items():
            if k not in ("epoch", "step", "seconds"):
                tb.add_scalar(f"epoch/{k}", v, epoch + 1)
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
        torch.save({"state_dict": fusion.state_dict(), "channels": chans, "align": a.align, "epoch": epoch + 1}, a.output / f"fusion_epoch{epoch + 1}.pt")
        if a.limit_iters and step >= a.limit_iters:
            break
    torch.save({"state_dict": fusion.state_dict(), "channels": chans, "align": a.align, "epoch": epoch + 1}, a.output / "fusion_last.pt")
    tb.close(); log.close()
    print("Training of the temporal fusion is done", flush=True)


if __name__ == "__main__":
    main()
