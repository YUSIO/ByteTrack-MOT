"""Exp120: fine-tune YOLOX jointly with the slot-prior injection (fctrack.prior_inject) on frames whose slots were written
by the real tracker (run_online.py --dump-slots; nothing is synthesised and no annotated target is hidden).

Each item is one frame: the letterboxed image, its ground-truth boxes, and the heatmap of the slots the tracker had before
that frame (fctrack.slot_prior.render). Augmentation: horizontal flip (image, boxes and slots together). Loss = head loss
with the prior + head loss without it (so the no-prior path stays a detector); --no-inject trains the same detector the
same way without the module (the control for "one more round of training"). BatchNorm statistics stay frozen.
Exp122: --freeze-detector trains only the injection (the detector's weights never change, so the no-prior path is the
original detector), with loss = head loss with the prior; --bound limits the residual. Every step checks the loss and the
gradients: a non-finite step is skipped and counted, and training stops (exit 3) when more than 1 % of the steps after the
first 200 were skipped or the fp16 gradient scale falls below 1.

Usage (code root on PYTHONPATH):
  python fctrack/train_prior.py -f <exp file> -c <ckpt> --data <data.json> --prior-config <prior.json> --output <dir> [--no-inject]
data.json: [{"view": <MOT view root>/train, "slots": <run>/slots/<arm>, "sequences": [...]} ...]
Writes <output>/last.pt ({"model", "inj", "channels", "prior", "epoch"}), train_log.jsonl and TensorBoard events in <output>/tb.
"""
import argparse
import configparser
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from fctrack.prior_inject import PriorInjection, PriorYOLOX
from fctrack.slot_prior import render
from fctrack.weaken import weaken
from yolox.data.data_augment import preproc
from yolox.exp import get_exp

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def seq_info(d):
    ini = configparser.ConfigParser()
    ini.read(d / "seqinfo.ini")
    s = {k.lower(): v for k, v in ini["Sequence"].items()}
    return int(s["seqlength"]), int(s["imwidth"]), int(s["imheight"])


def load_slots(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        out[r["f"]] = [{"kind": k, "id": i, "gap": g, "side": sd, "pred": {n: tuple(v) for n, v in p.items()}} for k, i, g, sd, p in r["s"]]
    return out


class SlotFrames(torch.utils.data.Dataset):
    def __init__(self, entries, prior_cfg, size, max_labels=120, flip=0.5, seed=0, weaken_cfg=None):
        self.size, self.prior_cfg, self.max_labels, self.flip, self.seed, self.epoch = size, prior_cfg, max_labels, flip, seed, 0
        self.weaken_cfg = weaken_cfg  # Exp121: slot-anchored weakening of targets (fctrack.weaken); None = off
        self.items, self.meta, self.gt, self.slots = [], {}, {}, {}
        for e in entries:
            view, sdir = Path(e["view"]), Path(e["slots"])
            for s in e["sequences"]:
                key = (str(view), s)
                L, w, h = seq_info(view / s)
                self.meta[key] = (view / s, w, h)
                gt = np.loadtxt(view / s / "gt" / "gt.txt", delimiter=",", ndmin=2)
                gt = gt[gt[:, 6] >= 1] if gt.size else np.zeros((0, 9))
                self.gt[key] = {int(f): gt[gt[:, 0] == f][:, 2:6] for f in np.unique(gt[:, 0])}
                self.slots[key] = load_slots(sdir / f"{s}.jsonl")
                self.items += [(key, f) for f in range(1, L + 1)]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        key, f = self.items[i]
        d, w, h = self.meta[key]
        img = cv2.imread(str(d / "img1" / f"{f:06d}.jpg"))
        assert img is not None, (key, f)
        boxes = self.gt[key].get(f, np.zeros((0, 4))).copy()
        rng = np.random.default_rng([self.seed, self.epoch, i])
        flip = rng.random() < self.flip
        wk = np.zeros(3, np.float32)
        if self.weaken_cfg:
            img = img.copy()
            st = weaken(img, self.slots[key].get(f, []), boxes, self.weaken_cfg, np.random.default_rng([self.seed, self.epoch, i, 1]), self.prior_cfg["lost_order"])
            wk[:] = (st["pos"], st["neg_lost"], st["neg_bg"])
        if flip:
            img = img[:, ::-1]
            boxes[:, 0] = w - boxes[:, 0] - boxes[:, 2]
        x, r = preproc(np.ascontiguousarray(img), self.size, MEAN, STD)
        prior = render(self.slots[key].get(f, []), self.prior_cfg, r, self.size, flip_width=w if flip else None)
        x1 = np.clip(boxes[:, 0], 0, w); y1 = np.clip(boxes[:, 1], 0, h)
        x2 = np.clip(boxes[:, 0] + boxes[:, 2], 0, w); y2 = np.clip(boxes[:, 1] + boxes[:, 3], 0, h)
        keep = (x2 - x1 > 1) & (y2 - y1 > 1)
        lab = np.zeros((self.max_labels, 5), np.float32)
        b = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], 1)[keep] * r
        n = min(len(b), self.max_labels)
        lab[:n, 1:] = b[:n]
        return torch.from_numpy(x), torch.from_numpy(prior), torch.from_numpy(lab), torch.from_numpy(wk)


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--exp-file", required=True)
    ap.add_argument("-c", "--ckpt", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--prior-config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--no-inject", action="store_true")
    ap.add_argument("--freeze-detector", action="store_true", help="Exp122: train the injection only")
    ap.add_argument("--bound", type=float, default=None, help="Exp122: bound of the injection residual")
    ap.add_argument("--weaken", type=Path, default=None, help="Exp121: json of the weakening augmentation (fctrack.weaken)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr-inject", type=float, default=1e-3)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--limit-iters", type=int, default=0, help="smoke test only")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    a.output.mkdir(parents=True, exist_ok=False)
    exp = get_exp(a.exp_file, None)
    base = exp.get_model()
    base.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"])
    base.head.use_l1 = True  # as in the last epochs of the original training (no-augmentation stage)
    size = tuple(exp.input_size)
    base.to(dev).eval()  # Exp122 fix: the channel probe below ran in train mode before and moved every BatchNorm's running statistics once
    with torch.no_grad():
        chans = [f.shape[1] for f in base.backbone(torch.zeros(1, 3, *size, device=dev))]
    inj = None if a.no_inject else PriorInjection(chans, prior_channels=2, bound=a.bound).to(dev)
    if a.freeze_detector:
        assert inj is not None
        for q in base.parameters():
            q.requires_grad_(False)
    model = PriorYOLOX(base, inj)
    prior_cfg = json.loads(a.prior_config.read_text())
    weaken_cfg = json.loads(a.weaken.read_text()) if a.weaken else None
    ds = SlotFrames(json.loads(a.data.read_text()), prior_cfg, size, seed=a.seed, weaken_cfg=weaken_cfg)
    groups = [] if a.freeze_detector else [{"params": list(base.backbone.parameters()), "lr": a.lr_backbone}, {"params": list(base.head.parameters()), "lr": a.lr_head}]
    if inj is not None:
        groups.append({"params": list(inj.parameters()), "lr": a.lr_inject})
    opt = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    steps = a.epochs * ((len(ds) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=a.amp)
    tb = SummaryWriter(a.output / "tb")
    log = open(a.output / "train_log.jsonl", "w")
    meta = {"args": {k: str(v) for k, v in vars(a).items()}, "channels": chans, "frames": len(ds), "steps": steps, "prior": prior_cfg, "weaken": weaken_cfg,
            "trainable_parameters": {"detector": sum(p.numel() for p in base.parameters()), "inject": sum(p.numel() for p in inj.parameters()) if inj else 0}}
    (a.output / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta), flush=True)
    step, started, skipped = 0, time.time(), 0
    params = [q for q in model.parameters() if q.requires_grad]
    for epoch in range(a.epochs):
        ds.set_epoch(epoch)
        loader = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True,
                                             generator=torch.Generator().manual_seed(a.seed * 1000 + epoch))
        model.train(); freeze_bn(model)
        sums, iters = {}, 0
        for x, prior, lab, wk in loader:
            x, prior, lab = x.to(dev, non_blocking=True), prior.to(dev, non_blocking=True), lab.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp):
                if a.freeze_detector:  # the detector is a fixed function: only the injection and its effect through the head are differentiated
                    with torch.no_grad():
                        f = list(base.backbone(x))
                    out = {"prior": base.head(inj(f, prior), lab, x)}
                else:
                    out = model.losses(x, lab, prior if inj is not None else None)
            loss = sum(v[0] for v in out.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            finite = bool(torch.isfinite(loss)) and all(bool(torch.isfinite(q.grad).all()) for q in params if q.grad is not None)
            if finite:
                torch.nn.utils.clip_grad_norm_(params, 10.0)
            else:
                skipped += 1
            scaler.step(opt); scaler.update(); sched.step(); step += 1; iters += 1
            if step > 200 and (skipped > 0.01 * step or (a.amp and scaler.get_scale() < 1.0)):
                print(json.dumps({"abort": "non-finite steps", "step": step, "skipped": skipped, "grad_scale": scaler.get_scale()}), flush=True)
                tb.close(); log.close()
                raise SystemExit(3)
            vals = {"total": float(loss.detach()), "prior_share": float((prior.amax((1, 2, 3)) > 0).float().mean()),
                    "weaken/pos": float(wk[:, 0].sum()), "weaken/neg_lost": float(wk[:, 1].sum()), "weaken/neg_bg": float(wk[:, 2].sum())}
            for path, (tot, iou_l, conf_l, cls_l, l1_l, nfg) in out.items():
                vals.update({f"{path}/total": float(tot), f"{path}/iou": float(iou_l), f"{path}/conf": float(conf_l), f"{path}/l1": float(l1_l), f"{path}/num_fg": float(nfg)})
            for k, v in vals.items():
                sums[k] = sums.get(k, 0.0) + v
            if step % 20 == 0:
                for k, v in vals.items():
                    tb.add_scalar(f"train/{k}", v, step)
                for gi, g in enumerate(opt.param_groups):
                    tb.add_scalar(f"lr/group{gi}", g["lr"], step)
                if inj is not None:
                    tb.add_scalar("inject/last_weight_norm", float(sum(m_[-1].weight.norm() for m_ in inj.fuse)), step)
                tb.add_scalar("train/skipped_steps", skipped, step)
                tb.add_scalar("train/grad_scale", scaler.get_scale() if a.amp else 1.0, step)
            if a.limit_iters and step >= a.limit_iters:
                break
        n = max(1, iters)
        rec = {"epoch": epoch + 1, "step": step, "seconds": round(time.time() - started, 1), "skipped_steps": skipped,
               "grad_scale": scaler.get_scale() if a.amp else 1.0, **{k: v / n for k, v in sums.items()}}
        for k, v in rec.items():
            if k not in ("epoch", "step", "seconds", "skipped_steps", "grad_scale"):
                tb.add_scalar(f"epoch/{k}", v, epoch + 1)
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
        if a.limit_iters and step >= a.limit_iters:
            break
    torch.save({"model": base.state_dict(), "inj": inj.state_dict() if inj is not None else None, "channels": chans, "prior": prior_cfg,
                "bound": a.bound, "frozen_detector": a.freeze_detector, "skipped_steps": skipped, "epoch": epoch + 1}, a.output / "last.pt")
    tb.close(); log.close()
    print("Training of the slot-prior detector is done", flush=True)


if __name__ == "__main__":
    main()
