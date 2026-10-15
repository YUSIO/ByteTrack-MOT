"""Stage 2 of Exp071: fine-tune the detector and one track head together, or with the head's gradient stopped.

Both arms see the same samples in the same order with the same schedule and seed. Every batch: B current frames and
their B template frames go through the backbone once; the YOLOX detection loss is computed on the current frames
(annotated boxes, no mosaic: letterbox and horizontal flip only); the track-head loss on the queries of those frames.
`joint`: the head's loss reaches the neck and backbone through the window and template features.
`stopgrad`: the features are detached before the head, so the detector is trained by the detection loss alone.
The detector is saved in the checkpoint format of tools/export_uavswarm_detections.py ({"model": state_dict}).
"""
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from jdt.detector import load_detector, normalise, objectness_level0
from jdt.head import TrackHead, loss as head_loss, window_features
from jdt.query import stack
from jdt.samples import TrackSamples, collate
from jdt.train_heads import unseen_frames, validate


def all_frames(seqs):
    out = {}
    for d in seqs:
        g = np.loadtxt(d / "gt/gt.txt", delimiter=",", ndmin=2)
        out[d.name] = sorted({int(f) for f in g[:, 0]})
    return out


def detector_param_groups(model, lr, wd):
    """YOLOX convention: weight decay on conv / linear weights only."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.ndim > 1 else no_decay).append(p)
    return [{"params": decay, "weight_decay": wd, "lr": lr}, {"params": no_decay, "weight_decay": 0.0, "lr": lr}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--arm", choices=("joint", "stopgrad"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    tr = cfg["joint"]
    cv2.setNumThreads(1)
    torch.manual_seed(tr["seed"])
    np.random.seed(tr["seed"])
    root = Path(cfg["dataset_train_dir"])
    held = set(cfg["held_out"])
    seqs = sorted(d for d in root.iterdir() if (d / "gt/gt.txt").exists())
    fit, val = [d for d in seqs if d.name not in held], [d for d in seqs if d.name in held]
    train_set = TrackSamples(fit, all_frames(fit), train=True, labels=True)
    val_set = TrackSamples(val, unseen_frames(cfg["detector"]["val_json"]), train=False, seed=tr["seed"])
    val_set.items = val_set.items[:: tr.get("val_every", 4)]
    if a.limit:
        train_set.items = train_set.items[: a.limit]
        val_set.items = val_set.items[: max(8, a.limit // 4)]
    a.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(a.device)
    _, det = load_detector(cfg["detector"]["exp_file"], cfg["detector"]["ckpt"], device)
    det.head.use_l1 = True  # as in the last (no-augmentation) epochs of the Exp043 schedule
    head = TrackHead(cfg["head"]["groups"]).to(device)
    head.load_state_dict(torch.load(cfg["head"]["init"], map_location=device)[cfg["head"]["key"]])
    g = torch.Generator()
    g.manual_seed(tr["seed"])
    loader = DataLoader(train_set, batch_size=tr["batch"], shuffle=True, generator=g, num_workers=tr["workers"], collate_fn=collate, drop_last=True, persistent_workers=True)
    vloader = DataLoader(val_set, batch_size=tr["batch"], shuffle=False, num_workers=tr["workers"], collate_fn=collate)
    opt_d = torch.optim.SGD(detector_param_groups(det, tr["det_lr"], tr["det_wd"]), momentum=0.9, nesterov=True)
    opt_h = torch.optim.AdamW(head.parameters(), lr=tr["head_lr"], weight_decay=1e-4)
    total = tr["epochs"] * len(loader)
    warm = tr["warmup_iters"]
    scale = lambda it: (it + 1) / warm if it < warm else 0.5 * (1 + math.cos(math.pi * (it - warm) / max(1, total - warm)))
    scaler = torch.amp.GradScaler("cuda")
    info = {"arm": a.arm, "fit_sequences": [d.name for d in fit], "fit_items": len(train_set), "held_out_items": len(val_set), "iterations": total, "torch": torch.__version__}
    print(json.dumps(info), flush=True)
    log, t0, it = [], time.time(), 0
    for ep in range(tr["epochs"]):
        det.train()
        head.train()
        acc = {"det": 0.0, "head": 0.0, "n": 0}
        for imgs, qs, qimg, B, tg in loader:
            for gr in opt_d.param_groups:
                gr["lr"] = tr["det_lr"] * scale(it)
            for gr in opt_h.param_groups:
                gr["lr"] = tr["head_lr"] * scale(it)
            x = normalise(imgs.to(device, non_blocking=True))
            tg = tg.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                fpn = det.backbone(x)
                dl = det.head([f[:B] for f in fpn], tg, x[:B])[0]
                p3 = fpn[0]
                obj = objectness_level0(det.head, p3)
                if a.arm == "stopgrad":
                    p3, obj = p3.detach(), obj.detach()
            if qs:
                q = stack(qs, device)
                qi = qimg.to(device)
                fs, ob, tf = window_features(p3.float(), obj.float(), qi, q["centre"], q["half"], p3.float(), q["tbox"], qi + B)
                hl, _ = head_loss(head(fs, ob, tf, q), q)
            else:
                hl = torch.zeros((), device=device)
            lossv = dl + tr["head_weight"] * hl
            opt_d.zero_grad(set_to_none=True)
            opt_h.zero_grad(set_to_none=True)
            scaler.scale(lossv).backward()
            scaler.step(opt_d)
            scaler.step(opt_h)
            scaler.update()
            acc["det"] += float(dl)
            acc["head"] += float(hl)
            acc["n"] += 1
            it += 1
            if it % 200 == 0:
                print(json.dumps({"iter": it, "det_loss": acc["det"] / acc["n"], "head_loss": acc["head"] / acc["n"], "seconds": int(time.time() - t0),
                                  "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}), flush=True)
        det.eval()
        rec = {"epoch": ep + 1, "seconds": int(time.time() - t0), "det_loss": acc["det"] / max(1, acc["n"]), "head_loss": acc["head"] / max(1, acc["n"]),
               "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
        rec["held_out"] = validate(det, {"all": head}, vloader, device)["all"]
        log.append(rec)
        print(json.dumps(rec), flush=True)
        torch.save({"model": det.state_dict(), "start_epoch": ep + 1}, a.output / "detector_last.pth.tar")
        torch.save({cfg["head"]["key"]: head.state_dict()}, a.output / "head_last.pt")
    (a.output / "train_log.json").write_text(json.dumps({"info": info, "config": cfg, "epochs": log}, indent=1) + "\n")


if __name__ == "__main__":
    main()
