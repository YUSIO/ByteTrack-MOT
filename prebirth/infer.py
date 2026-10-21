"""Causal, image-only dense proposals for Exp138. Never reads annotations.

Run as ``python -m prebirth.infer``. ``split=train/dev`` means the first/second
half of each official training sequence; ``full`` processes the supplied view.
By default exported frame indices start at one within the selected split, so a
dev export can be consumed with the corresponding reindexed MOT dev view.
"""
import argparse
import configparser
import hashlib
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sequence_root(root):
    root = Path(root).resolve()
    if (root / "train").is_dir():
        return root / "train"
    return root


def read_sequence_info(path):
    ini = configparser.ConfigParser()
    if not ini.read(path / "seqinfo.ini") or "Sequence" not in ini:
        raise ValueError(f"Missing Sequence metadata: {path}")
    section = ini["Sequence"]
    return {"length": int(section["seqLength"]),
            "width": int(section["imWidth"]), "height": int(section["imHeight"]),
            "image_dir": section.get("imDir", "img1"),
            "extension": section.get("imExt", ".jpg")}


def frame_bounds(length, split):
    if split == "train":
        bounds = (1, length // 2)
    elif split == "dev":
        bounds = (length // 2 + 1, length)
    else:
        bounds = (1, length)
    if bounds[1] < bounds[0]:
        raise ValueError(f"Empty {split} split for sequence length {length}")
    return bounds


def tile_starts(length, size, overlap):
    if size <= 0 or not 0 <= overlap < size or length <= 0:
        raise ValueError("Require positive dimensions and 0 <= overlap < tile size")
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, size - overlap))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def tile_coordinates(width, height, size, overlap):
    return [(x, y, min(size, width - x), min(size, height - y))
            for y in tile_starts(height, size, overlap)
            for x in tile_starts(width, size, overlap)]


def iou_xyxy(box, boxes):
    if not len(boxes):
        return np.empty(0, dtype=np.float64)
    lo = np.maximum(box[:2], boxes[:, :2])
    hi = np.minimum(box[2:], boxes[:, 2:])
    inter = np.maximum(hi - lo, 0).prod(axis=1)
    area = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
    other = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
    return inter / np.maximum(area + other - inter, 1e-12)


def nms(rows, threshold):
    """Rows are x1,y1,x2,y2,score; stable ties make tile dedup reproducible."""
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, 5)
    order = np.argsort(-rows[:, 4], kind="stable")
    keep = []
    while len(order):
        i = int(order[0])
        keep.append(i)
        order = order[1:][iou_xyxy(rows[i, :4], rows[order[1:], :4]) < threshold]
    return rows[keep]


def decode_batch(outputs, coordinates, image_shape, threshold, topk_per_tile,
                 tile_size, min_box_size, max_box_size):
    """Decode raw offset residuals (no sigmoid); scores alone use sigmoid."""
    heat = outputs["heatmap"].sigmoid()
    if heat.ndim != 4 or heat.shape[1] != 1:
        raise ValueError(f"Unexpected heatmap shape {tuple(heat.shape)}")
    if outputs["offset"].shape != (len(coordinates), 2, *heat.shape[-2:]):
        raise ValueError("Offset/heatmap shapes do not agree")
    if outputs["log_wh"].shape != outputs["offset"].shape:
        raise ValueError("Box/offset shapes do not agree")
    maxima = heat.eq(F.max_pool2d(heat, 3, stride=1, padding=1))
    height, width = image_shape
    result, counts = [], {"threshold_localmax": 0, "tile_budget_candidates": 0,
                          "invalid_or_padding": 0, "decoded": 0}
    max_size = max_box_size if max_box_size else tile_size
    for b, (left, top, valid_w, valid_h) in enumerate(coordinates):
        mask = maxima[b, 0] & (heat[b, 0] >= threshold)
        ys, xs = torch.where(mask)
        counts["threshold_localmax"] += len(xs)
        scores = heat[b, 0, ys, xs]
        if not len(scores):
            continue
        offset = outputs["offset"][b, :, ys, xs].T
        log_wh = outputs["log_wh"][b, :, ys, xs].T
        center = (torch.stack((xs, ys), dim=1).to(offset.dtype) + offset) * 4.0
        wh = log_wh.clamp(math.log(min_box_size), math.log(max_size)).exp()
        finite = torch.isfinite(center).all(1) & torch.isfinite(log_wh).all(1)
        valid = (finite & (center[:, 0] >= 0) & (center[:, 0] < valid_w)
                 & (center[:, 1] >= 0) & (center[:, 1] < valid_h))
        counts["invalid_or_padding"] += int((~valid).sum().item())
        center = center[valid] + center.new_tensor((left, top))
        wh, scores = wh[valid], scores[valid]
        boxes = torch.cat((center - wh / 2, center + wh / 2), dim=1)
        boxes[:, 0::2].clamp_(0, width)
        boxes[:, 1::2].clamp_(0, height)
        good = ((boxes[:, 2] - boxes[:, 0]) >= min_box_size) & ((boxes[:, 3] - boxes[:, 1]) >= min_box_size)
        counts["invalid_or_padding"] += int((~good).sum().item())
        boxes, scores = boxes[good], scores[good]
        if len(scores) > topk_per_tile:
            selected = torch.argsort(scores, descending=True, stable=True)[:topk_per_tile]
            boxes, scores = boxes[selected], scores[selected]
        counts["tile_budget_candidates"] += len(scores)
        rows = torch.cat((boxes, scores[:, None]), dim=1).detach().cpu().numpy()
        counts["decoded"] += len(rows)
        result.extend(rows)
    return np.asarray(result, dtype=np.float64).reshape(-1, 5), counts


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def load_model(checkpoint, device):
    from prebirth.model import TemporalDetector
    # Local, explicitly supplied checkpoint; weights_only avoids arbitrary pickle execution.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or "model" not in saved or "model_config" not in saved:
        raise ValueError("Checkpoint requires model and model_config keys")
    config = saved["model_config"]
    model = TemporalDetector(**config)
    model.load_state_dict(saved["model"], strict=True)
    training = {"args": saved.get("args", {}), "epoch": saved.get("epoch"),
                "best_epoch": saved.get("best_epoch"), "best_dev_loss": saved.get("best_dev_loss")}
    return model.to(device).eval(), config, training


def export_sequence(model, directory, output, args, device):
    info = read_sequence_info(directory)
    start, stop = frame_bounds(info["length"], args.split)
    tiles = tile_coordinates(info["width"], info["height"], args.tile_size, args.overlap)
    cache = OrderedDict()

    def read_frame(index):
        if index not in cache:
            path = directory / info["image_dir"] / f"{index:06d}{info['extension']}"
            raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if raw is None:
                raise FileNotFoundError(f"Cannot decode image: {path}")
            if raw.shape[:2] != (info["height"], info["width"]):
                raise ValueError(f"Image/seqinfo size mismatch: {path}: {raw.shape}")
            cache[index] = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        return cache[index]

    output.mkdir(parents=True, exist_ok=False)
    det_file = output / "det.txt"
    stats_path = output / "frame_stats.jsonl"
    totals = {"frames": 0, "tiles": 0, "threshold_localmax": 0,
              "tile_budget_candidates": 0, "invalid_or_padding": 0, "decoded": 0,
              "after_nms": 0, "exported": 0, "model_seconds": 0.0}
    begin = time.perf_counter()
    with det_file.open("w") as det, stats_path.open("w") as stats:
        for frame in range(start, stop + 1):
            history = [max(start, frame - j * args.history_gap) for j in range(3)]
            images = [read_frame(index) for index in history]
            frame_stats = {"frame": frame if args.frame_numbering == "original" else frame - start + 1,
                           "source_frame": frame, "history_frames": history,
                           "distinct_observed_frames": len(set(history)), "tiles": len(tiles),
                           "threshold_localmax": 0, "tile_budget_candidates": 0,
                           "invalid_or_padding": 0, "decoded": 0, "model_seconds": 0.0}
            decoded = []
            for offset in range(0, len(tiles), args.batch_size):
                coords = tiles[offset:offset + args.batch_size]
                # Padding is zero in RGB [0,1]; valid pixel support is recorded above.
                batch = np.zeros((len(coords), 3, args.tile_size, args.tile_size, 3), dtype=np.uint8)
                for b, (x, y, w, h) in enumerate(coords):
                    for t, img in enumerate(images):
                        batch[b, t, :h, :w] = img[y:y+h, x:x+w]
                tensor = torch.from_numpy(batch).permute(0, 1, 4, 2, 3).to(device=device, dtype=torch.float32) / 255.0
                sync(device)
                tick = time.perf_counter()
                with torch.inference_mode():
                    predictions = model(tensor)
                sync(device)
                frame_stats["model_seconds"] += time.perf_counter() - tick
                rows, counts = decode_batch(predictions, coords,
                    (info["height"], info["width"]), args.threshold, args.topk_per_tile,
                    args.tile_size, args.min_box_size, args.max_box_size)
                decoded.extend(rows)
                for key, value in counts.items():
                    frame_stats[key] += value
            rows = nms(decoded, args.nms_iou)
            frame_stats["after_nms"] = len(rows)
            rows = rows[:args.topk]
            frame_stats["exported"] = len(rows)
            for x1, y1, x2, y2, score in rows:
                # repr preserves the decoded floating-point score; no score calibration here.
                values = [frame_stats["frame"], -1, float(x1), float(y1),
                          float(x2 - x1), float(y2 - y1), float(score), -1, -1, -1]
                det.write(",".join(str(v) for v in values) + "\n")
            stats.write(json.dumps(frame_stats, allow_nan=False) + "\n")
            totals["frames"] += 1
            for key in totals:
                if key != "frames":
                    totals[key] += frame_stats[key]
            # Retain only frames needed by a future causal window (bounded memory).
            for old in list(cache):
                if old < frame - 2 * args.history_gap:
                    del cache[old]
    return {**info, "source_frame_start": start, "source_frame_stop": stop,
            "frame_numbering": args.frame_numbering,
            "spatial_coverage_fraction": 1.0, "tiles_per_frame": len(tiles),
            "coverage_definition": "All original image pixels covered; no annotation-based recall is measured",
            "seqinfo_sha256": sha256(directory / "seqinfo.ini"),
            "totals": totals, "wall_seconds": time.perf_counter() - begin,
            "det_sha256": sha256(det_file), "frame_stats_sha256": sha256(stats_path)}


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", choices=("train", "dev", "full"), default="dev")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--topk", type=int, default=100, help="Maximum exported candidates per frame after tile NMS")
    ap.add_argument("--topk-per-tile", type=int, default=128)
    ap.add_argument("--history-gap", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--min-box-size", type=float, default=2.0)
    ap.add_argument("--max-box-size", type=float, default=0, help="0 clips each decoded dimension to tile size")
    ap.add_argument("--frame-numbering", choices=("split", "original"), default="split")
    ap.add_argument("--allow-input-mismatch", action="store_true",
                    help="Explicitly permit and record a tile size/history gap different from training")
    ap.add_argument("--sequences", nargs="+", help="Optional sequence names, space or comma separated")
    return ap


def main():
    args = parser().parse_args()
    if (args.tile_size % 4 or args.tile_size <= 0 or not 0 <= args.overlap < args.tile_size
            or args.topk <= 0 or args.topk_per_tile <= 0 or args.batch_size <= 0
            or args.history_gap <= 0 or not 0 <= args.threshold <= 1
            or not 0 < args.nms_iou <= 1 or args.min_box_size <= 0
            or (args.max_box_size and args.max_box_size < args.min_box_size)):
        raise ValueError("Invalid inference dimensions, budgets, thresholds or history gap")
    data = sequence_root(args.data_root)
    names = sorted(path.name for path in data.glob("UAVSwarm-*") if path.is_dir())
    if args.sequences:
        requested = {name for item in args.sequences for name in item.split(",")}
        missing = requested - set(names)
        if missing:
            raise ValueError(f"Requested sequences not found: {sorted(missing)}")
        names = [name for name in names if name in requested]
    if not names:
        raise ValueError(f"No UAVSwarm sequences under {data}")
    device = torch.device(args.device)
    model, model_config, training = load_model(args.checkpoint, device)
    mismatch = {}
    for saved_key, requested_key in (("patch_size", "tile_size"), ("history_gap", "history_gap")):
        saved_value = training["args"].get(saved_key)
        requested_value = getattr(args, requested_key)
        if saved_value is not None and saved_value != requested_value:
            mismatch[requested_key] = {"training": saved_value, "inference": requested_value}
    if mismatch and not args.allow_input_mismatch:
        raise ValueError(f"Training/inference input mismatch {mismatch}; use matching inputs or --allow-input-mismatch")
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = {"status": "running", "command": sys.argv, "data_root": str(data),
                "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
                "model_config": model_config, "model_source_sha256": sha256(Path(__file__).with_name("model.py")),
                "checkpoint_training": training, "training_input_mismatch": mismatch,
                "inference_source_sha256": sha256(__file__),
                "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "torch_version": str(torch.__version__), "numpy_version": np.__version__,
                "annotation_access": False, "history_order": "current,t-gap,t-2gap",
                "padding": "RGB zero; centers outside original tile support are discarded",
                "score_semantics": "sigmoid heatmap, uncalibrated, unchanged in export",
                "sequences": {}}
    write_json(args.out / "metadata.json", metadata)
    try:
        for name in names:
            result = export_sequence(model, data / name, args.out / name, args, device)
            metadata["sequences"][name] = result
            write_json(args.out / "metadata.json", metadata)
            print(name, json.dumps(result["totals"]), flush=True)
        metadata["status"] = "completed"
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(args.out / "metadata.json", metadata)


if __name__ == "__main__":
    main()
