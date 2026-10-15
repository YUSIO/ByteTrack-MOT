"""Export a per-sequence detection cache from a trained YOLOX checkpoint (Exp043).

Reads images only; never reads ground truth. Each sequence gets det.txt with rows
`frame,-1,x,y,w,h,score,-1,-1,-1` in original-image pixels, score = objectness * class
confidence. Boxes are neither clipped nor filtered beyond the confidence/NMS thresholds.
Works on cuda, mps or cpu: the head output is decoded here instead of via the
`.type(dtype)` path of YOLOXHead.decode_outputs, which is CUDA-oriented.
"""
import argparse
import configparser
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import postprocess

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def decode(raw, hw, strides):
    """Same arithmetic as YOLOXHead.decode_outputs."""
    grids, expanded = [], []
    for (h, w), stride in zip(hw, strides):
        yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing="ij")
        grids.append(torch.stack((xv, yv), 2).view(1, -1, 2))
        expanded.append(torch.full((1, h * w, 1), float(stride)))
    grids = torch.cat(grids, dim=1).to(raw)
    expanded = torch.cat(expanded, dim=1).to(raw)
    out = raw.clone()
    out[..., :2] = (raw[..., :2] + grids) * expanded
    out[..., 2:4] = torch.exp(raw[..., 2:4]) * expanded
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-f", "--exp-file", required=True)
    p.add_argument("-c", "--ckpt", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True, help="MOT root containing train/ and test/")
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--sequences", nargs="+")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--nms", type=float, default=0.7)
    p.add_argument("--tsize", type=int, default=1088)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    a.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(a.device)
    exp = get_exp(a.exp_file, None)
    model = exp.get_model().eval()
    state = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.head.decode_in_inference = False
    model.to(device)
    size = (a.tsize, a.tsize)
    names = a.sequences or sorted(x.name for x in (a.data_root / a.split).glob("UAVSwarm-*") if x.is_dir())
    summary = {"checkpoint": str(a.ckpt), "checkpoint_sha256": sha256(a.ckpt), "checkpoint_epoch": int(state.get("start_epoch", -1)),
               "exp_file": a.exp_file, "split": a.split, "test_size": list(size), "conf": a.conf, "nms": a.nms,
               "device": a.device, "batch": a.batch, "precision": "fp32", "tta": False, "gt_read": False,
               "torch": torch.__version__, "sequences": []}
    for name in names:
        seq = a.data_root / a.split / name
        ini = configparser.ConfigParser()
        ini.read(seq / "seqinfo.ini")
        info = ini["Sequence"]
        length, img_dir, ext = int(info["seqLength"]), info.get("imDir", "img1"), info.get("imExt", ".jpg")
        rows, started = [], time.time()
        for start in range(1, length + 1, a.batch):
            frames = list(range(start, min(start + a.batch, length + 1)))
            tensors, ratios = [], []
            for frame in frames:
                image = cv2.imread(str(seq / img_dir / f"{frame:06d}{ext}"))
                assert image is not None, (name, frame)
                x, r = preproc(image, size, MEAN, STD)
                tensors.append(torch.from_numpy(x))
                ratios.append(r)
            with torch.inference_mode():
                raw = model(torch.stack(tensors).to(device)).float().cpu()
                outputs = postprocess(decode(raw, model.head.hw, model.head.strides), exp.num_classes, a.conf, a.nms)
            for frame, r, out in zip(frames, ratios, outputs):
                if out is None:
                    continue
                boxes = out[:, :4].numpy() / r
                scores = (out[:, 4] * out[:, 5]).numpy()
                for (x1, y1, x2, y2), s in zip(boxes, scores):
                    rows.append(f"{frame},-1,{x1:.4f},{y1:.4f},{x2 - x1:.4f},{y2 - y1:.4f},{s:.6f},-1,-1,-1\n")
        (a.output / name).mkdir()
        path = a.output / name / "det.txt"
        path.write_text("".join(rows))
        summary["sequences"].append({"sequence": name, "frames": length, "detections": len(rows),
                                     "seconds": round(time.time() - started, 2), "det_sha256": sha256(path)})
        print(json.dumps(summary["sequences"][-1]), flush=True)
    summary["total_frames"] = sum(s["frames"] for s in summary["sequences"])
    summary["total_detections"] = sum(s["detections"] for s in summary["sequences"])
    (a.output / "export_summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
