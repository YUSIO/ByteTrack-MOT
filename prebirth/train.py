"""Train the causal dense pre-birth detector, selecting on fixed train/dev halves.

Run as ``python -m prebirth.train --root DATA --out NEW_RUN/model``. Test GT is
never read. Validation uses fixed native-resolution patches and is a selection
proxy, not a replacement for full-sequence detector/MOT evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from prebirth.data import TemporalPatchDataset, collate_patches, seed_worker, sha256
from prebirth.model import TemporalDetector


def write_json(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def detection_loss(output, batch, offset_weight=1.0, wh_weight=0.1):
    logits = output["heatmap"].float()
    target, valid = batch["heatmap"].float(), batch["hm_valid"].float()
    if logits.shape != target.shape:
        raise ValueError(f"Heatmap shape mismatch: {tuple(logits.shape)} vs {tuple(target.shape)}")
    prob = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    positive = target.eq(1).float() * valid
    negative = target.lt(1).float() * valid
    positive_loss = -(1 - prob).pow(2) * prob.log() * positive
    negative_loss = -prob.pow(2) * (1 - prob).log() * (1 - target).pow(4) * negative
    heatmap_loss = (positive_loss.sum() + negative_loss.sum()) / positive.sum().clamp_min(1)
    mask = batch["reg_mask"].float()
    divisor = (2 * mask.sum()).clamp_min(1)
    offset_loss = ((output["offset"].float() - batch["offset"]).abs() * mask).sum() / divisor
    wh_loss = ((output["log_wh"].float() - batch["log_wh"]).abs() * mask).sum() / divisor
    loss = heatmap_loss + offset_weight * offset_loss + wh_weight * wh_loss
    return loss, {"loss": loss.detach(), "heatmap_loss": heatmap_loss.detach(),
                  "offset_loss": offset_loss.detach(), "log_wh_loss": wh_loss.detach()}


def box_iou(boxes, ground_truth):
    lt = torch.maximum(boxes[:, None, :2], ground_truth[None, :, :2])
    rb = torch.minimum(boxes[:, None, 2:], ground_truth[None, :, 2:])
    overlap = (rb - lt).clamp_min(0).prod(2)
    area_a = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(1)
    area_b = (ground_truth[:, 2:] - ground_truth[:, :2]).clamp_min(0).prod(1)
    return overlap / (area_a[:, None] + area_b[None] - overlap).clamp_min(1e-6)


@torch.no_grad()
def patch_proxy(output, batch, thresholds=(0.1, 0.3), topk=100):
    """Fixed local-max decoding + IoU50 matching; no GT enters candidate decoding."""
    heatmap = output["heatmap"].float().sigmoid()
    peak = heatmap * heatmap.eq(F.max_pool2d(heatmap, 3, 1, 1))
    # Ignore padded/marked-label regions in this diagnostic metric, not in inference.
    peak = peak * batch["hm_valid"]
    b, _, h, w = peak.shape
    scores, indices = peak.flatten(1).topk(min(topk, h * w), dim=1)
    offset = output["offset"].float().flatten(2)
    log_wh = output["log_wh"].float().flatten(2)
    gathered_offset = offset.gather(2, indices[:, None].expand(-1, 2, -1)).transpose(1, 2)
    wh = log_wh.gather(2, indices[:, None].expand(-1, 2, -1)).transpose(1, 2).clamp(-3, 12).exp()
    center = torch.stack((indices % w, indices // w), dim=2).float()
    center = (center + gathered_offset) * 4
    boxes = torch.cat((center - wh / 2, center + wh / 2), dim=2).cpu()
    scores = scores.cpu()
    result = {str(t): {"tp": 0, "fp": 0, "fn": 0} for t in thresholds}
    for i in range(b):
        gt = batch["boxes"][i].cpu()
        for threshold in thresholds:
            predicted = boxes[i, scores[i] >= threshold]
            ious = box_iou(predicted, gt)
            used = torch.zeros(len(gt), dtype=torch.bool)
            tp = 0
            for row in ious:
                if not len(gt):
                    break
                row = row.masked_fill(used, -1)
                score, index = row.max(0)
                if score >= 0.5:
                    used[index] = True
                    tp += 1
            counts = result[str(threshold)]
            counts["tp"] += tp
            counts["fp"] += len(predicted) - tp
            counts["fn"] += len(gt) - tp
    return result


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def autocast_context(args, device):
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype,
                          enabled=args.amp and device.type == "cuda")


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    totals, samples, collisions = {}, 0, 0
    counts = {str(t): {"tp": 0, "fp": 0, "fn": 0} for t in (0.1, 0.3)}
    for batch in loader:
        batch = to_device(batch, device)
        with autocast_context(args, device):
            output = model(batch["images"])
        _, losses = detection_loss(output, batch, args.offset_weight, args.wh_weight)
        n = len(batch["images"])
        samples += n
        collisions += batch["collisions"]
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value) * n
        proxy = patch_proxy(output, batch)
        for threshold, values in proxy.items():
            for key, value in values.items():
                counts[threshold][key] += value
    metrics = {key: value / samples for key, value in totals.items()}
    for values in counts.values():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        values.update(precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn),
                      f1=2 * tp / max(1, 2 * tp + fp + fn))
    metrics.update(samples=samples, center_cell_collisions=collisions, patch_iou50_proxy=counts)
    if not all(math.isfinite(metrics[k]) for k in totals):
        raise FloatingPointError(f"Non-finite validation loss: {metrics}")
    return metrics


def checkpoint(model, optimizer, scaler, scheduler, epoch, best_epoch, best_loss, args):
    numpy_state = np.random.get_state()
    numpy_state = [numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]),
                   int(numpy_state[3]), float(numpy_state[4])]
    return {"model": model.state_dict(), "model_config": model.model_config,
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch,
            "best_epoch": best_epoch, "best_dev_loss": best_loss,
            "args": vars(args), "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": numpy_state, "python_rng_state": random.getstate()}


def save_checkpoint(path, state):
    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", choices=["train"], default="train", help="Fitting always uses first train half; dev uses second")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["single", "fusion", "aligned"], default="aligned")
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--steps-per-epoch", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--patch-size", type=int, default=320)
    parser.add_argument("--history-gap", type=int, default=3)
    parser.add_argument("--dev-samples", type=int, default=720)
    parser.add_argument("--dev-batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=138)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--offset-weight", type=float, default=1.0)
    parser.add_argument("--wh-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16",
                        help="CUDA autocast precision; gradient scaling is used only with fp16")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--sequences", nargs="+", default=None, help="Optional preflight-only sequence subset")
    parser.add_argument("--hard-negative-json", default=None,
                        help="Verified train-half mining manifest replacing only the background-seeking quarter")
    parser.add_argument("--log-every", type=int, default=25)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--init-checkpoint", default=None, help="Model weights only; reset optimizer/epoch in a new run")
    initialization.add_argument("--resume", default=None, help="Restore training state into a NEW output directory")
    args = parser.parse_args()
    for key in ("epochs", "steps_per_epoch", "batch_size", "dev_samples", "dev_batch_size", "cpu_threads", "log_every"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.workers < 0 or args.lr <= 0 or args.grad_clip <= 0:
        parser.error("workers >= 0, lr > 0 and grad-clip > 0 are required")
    if args.sequences:
        args.sequences = [name for value in args.sequences for name in value.split(",") if name]
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device unavailable: {device}")
    if device.type == "cuda" and args.amp and args.amp_dtype == "bf16":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(f"Requested bf16 autocast unsupported by {device}")
    # Importing TensorBoard is a preflight dependency check, never silently omit events.
    from torch.utils.tensorboard import SummaryWriter
    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_set = TemporalPatchDataset(args.root, "train", args.steps_per_epoch * args.batch_size,
                                    args.patch_size, args.history_gap, args.seed, sequences=args.sequences,
                                    hard_negative_json=args.hard_negative_json)
    args.hard_negative_json_sha256 = train_set.hard_negative_info["sha256"] if train_set.hard_negative_info else None
    dev_set = TemporalPatchDataset(args.root, "dev", args.dev_samples, args.patch_size,
                                  args.history_gap, args.seed + 100000, augment=False, sequences=args.sequences)
    common = dict(num_workers=args.workers, collate_fn=collate_patches, worker_init_fn=seed_worker,
                  pin_memory=device.type == "cuda", persistent_workers=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=True, **common)
    dev_loader = DataLoader(dev_set, batch_size=args.dev_batch_size, shuffle=False, **common)
    model = TemporalDetector(mode=args.mode, channels=args.channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * args.steps_per_epoch
    warmup_steps = min(int(args.warmup_epochs * args.steps_per_epoch), total_steps - 1)

    def lr_factor(step):
        if step < warmup_steps:
            return max(0.01, (step + 1) / max(1, warmup_steps))
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    use_scaler = args.amp and device.type == "cuda" and args.amp_dtype == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    start_epoch = 0
    parent = args.resume or args.init_checkpoint
    if parent:
        state = torch.load(parent, map_location="cpu", weights_only=True)
        expected = model.model_config
        if state.get("model_config") != expected:
            raise ValueError(f"Checkpoint model config {state.get('model_config')} != {expected}")
        model.load_state_dict(state["model"], strict=True)
        if args.resume:
            for key in ("steps_per_epoch", "batch_size", "patch_size", "history_gap", "seed", "lr", "warmup_epochs", "epochs", "amp", "amp_dtype", "hard_negative_json_sha256"):
                if state["args"].get(key) != getattr(args, key):
                    raise ValueError(f"Resume changes {key}; use --init-checkpoint for a new training schedule")
            optimizer.load_state_dict(state["optimizer"])
            for values in optimizer.state.values():
                for key, value in values.items():
                    if isinstance(value, torch.Tensor):
                        values[key] = value.to(device)
            scaler.load_state_dict(state["scaler"])
            scheduler.load_state_dict(state["scheduler"])
            start_epoch = state["epoch"]
            torch.set_rng_state(state["torch_rng_state"])
            if device.type == "cuda" and state.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng_state"])
            numpy_state = state["numpy_rng_state"]
            np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                                 numpy_state[2], numpy_state[3], numpy_state[4]))
            random.setstate(state["python_rng_state"])
    if start_epoch >= args.epochs:
        raise ValueError("No remaining epochs; use --init-checkpoint to start further training")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), "selection": "minimum fixed-dev-patch total loss among epochs completed in this attempt",
              "precision": {"parameter_dtype": "float32",
                            "autocast_dtype": args.amp_dtype if args.amp and device.type == "cuda" else "float32",
                            "gradient_scaling": bool(use_scaler)},
              "sampling": ("50% target-offset, 25% random, 25% mined-hard-negative; all in-crop centers supervised"
                           if train_set.hard_negative_info else
                           "50% target-offset, 25% random, 25% background-seeking; all in-crop centers supervised"),
              "hard_negative_mining": train_set.hard_negative_info,
              "history_order": [0, -args.history_gap, -2 * args.history_gap],
              "source_xywh_coordinates": "as recorded, no +/-1 conversion", "stride": 4,
              "python_executable": sys.executable, "python_version": platform.python_version(),
              "torch_version": torch.__version__, "numpy_version": np.__version__, "opencv_version": cv2.__version__,
              "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
              "parent_checkpoint_sha256": sha256(Path(parent)) if parent else None,
              "parameters": sum(p.numel() for p in model.parameters())}
    write_json(out / "config.json", config)
    source = {"train": train_set.source_info, "dev": dev_set.source_info,
              "hard_negative_mining": train_set.hard_negative_info,
              "hard_negative_sampling_completed_epochs": [],
              "boundary_policy": "history frame=max(split first frame,current-k*gap), never crosses half boundary",
              "validation_sampling": {"seed": args.seed + 100000, "samples": args.dev_samples,
                                      "patch_size": args.patch_size, "epoch": 0}}
    write_json(out / "source_split.json", source)
    print(json.dumps({"event": "preflight_complete", "config": config,
                      "train_frames": len(train_set.frames), "dev_frames": len(dev_set.frames)}), flush=True)
    writer = SummaryWriter(str(out / "tb"))
    history, best_loss, best_epoch = [], float("inf"), None
    started = time.time()
    try:
        for epoch in range(start_epoch, args.epochs):
            train_set.set_epoch(epoch)
            model.train()
            totals, samples, collisions = {}, 0, 0
            hard_negative_samples, hard_negative_kinds = 0, {}
            epoch_started = time.time()
            for step, batch in enumerate(train_loader):
                batch = to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(args, device):
                    output = model(batch["images"])
                loss, values = detection_loss(output, batch, args.offset_weight, args.wh_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch={epoch + 1} step={step + 1}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                n = len(batch["images"])
                samples += n
                collisions += batch["collisions"]
                for sample_metadata in batch["metadata"]:
                    if "hard_negative_source_row" in sample_metadata:
                        hard_negative_samples += 1
                        kind = sample_metadata["hard_negative_kind"]
                        hard_negative_kinds[kind] = hard_negative_kinds.get(kind, 0) + 1
                for key, value in values.items():
                    totals[key] = totals.get(key, 0.0) + float(value) * n
                global_step = epoch * args.steps_per_epoch + step + 1
                if (step + 1) % args.log_every == 0 or step == 0:
                    row = {"event": "train_step", "epoch": epoch + 1, "step": step + 1,
                           "loss": float(loss.detach()), "lr": optimizer.param_groups[0]["lr"],
                           "gradient_norm": float(grad_norm), "elapsed_seconds": time.time() - epoch_started}
                    print(json.dumps(row), flush=True)
                    for key, value in values.items():
                        writer.add_scalar(f"train_step/{key}", float(value), global_step)
                    writer.add_scalar("train_step/lr", row["lr"], global_step)
            train_metrics = {key: value / samples for key, value in totals.items()}
            train_metrics.update(samples=samples, center_cell_collisions=collisions)
            if train_set.hard_negative_info:
                train_metrics.update(hard_negative_samples=hard_negative_samples,
                                     hard_negative_kind_counts=hard_negative_kinds)
                expected = train_set.hard_negative_info["expected_samples_per_epoch"]
                if hard_negative_samples != expected:
                    raise AssertionError(f"Hard-negative sample count {hard_negative_samples} != expected {expected}")
                source["hard_negative_sampling_completed_epochs"].append(
                    {"epoch": epoch + 1, "samples": hard_negative_samples, "kinds": hard_negative_kinds})
                write_json(out / "source_split.json", source)
            dev_metrics = validate(model, dev_loader, device, args)
            improved = dev_metrics["loss"] < best_loss
            if improved:
                best_loss, best_epoch = dev_metrics["loss"], epoch + 1
            record = {"epoch": epoch + 1, "train": train_metrics, "dev": dev_metrics,
                      "epoch_seconds": time.time() - epoch_started, "best_epoch": best_epoch,
                      "best_dev_loss": best_loss, "selected": improved}
            history.append(record)
            state = checkpoint(model, optimizer, scaler, scheduler, epoch + 1, best_epoch, best_loss, args)
            save_checkpoint(out / "last.pt", state)
            if improved:
                save_checkpoint(out / "best.pt", state)
            write_json(out / "history.json", history)
            metrics = {"status": "running" if epoch + 1 < args.epochs else "completed",
                       "completed_epochs_this_attempt": len(history), "last_epoch": epoch + 1,
                       "best_epoch": best_epoch, "best_dev_loss": best_loss,
                       "selection": config["selection"], "best_dev": next(r["dev"] for r in history if r["epoch"] == best_epoch),
                       "last_dev": dev_metrics, "last_train": train_metrics,
                       "elapsed_seconds": time.time() - started,
                       "evidence_boundary": "fixed GT-sampled dev patches; not full-frame detector or tracking metrics"}
            write_json(out / "metrics.json", metrics)
            for phase, values in (("train", train_metrics), ("dev", dev_metrics)):
                for key, value in values.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"{phase}/{key}", value, epoch + 1)
            for threshold, values in dev_metrics["patch_iou50_proxy"].items():
                for key, value in values.items():
                    writer.add_scalar(f"dev_proxy_at_{threshold}/{key}", value, epoch + 1)
            writer.flush()
            print(json.dumps({"event": "epoch_complete", **record}), flush=True)
    except BaseException as exc:
        write_json(out / "failure.json", {"error_type": type(exc).__name__, "message": str(exc),
                                          "completed_epochs": len(history), "elapsed_seconds": time.time() - started})
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    main()
