"""Export a per-sequence detection cache from a frozen YOLOX detector with the trained temporal fusion (Exp091).

Online: frame t fuses the previous frame's raw (unfused) FPN features; the first frame of a sequence is not fused.
Rows `frame,-1,x,y,w,h,score,-1,-1,-1` in original-image pixels, score = objectness x class confidence, as in
tools/export_uavswarm_detections.py. Reads images and camera-motion files only. `--no-fusion` exports the original
detector through the same code path (sanity check against an existing cache).
Usage (PYTHONPATH=.): python tfusion/export.py -f <exp> -c <det ckpt> --fusion <fusion_last.pt> --view <root>/train
       --sequences ... --gmc <dir> --output <dir>
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from tfusion.data import MEAN, STD, load_gmc, seq_info
from tfusion.module import TemporalFusion
from tfusion.train import frozen_detector
from yolox.data.data_augment import preproc
from yolox.utils import postprocess


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--exp-file", required=True)
    ap.add_argument("-c", "--ckpt", required=True)
    ap.add_argument("--fusion", type=Path)
    ap.add_argument("--no-fusion", action="store_true")
    ap.add_argument("--view", type=Path, required=True)
    ap.add_argument("--sequences", nargs="+", required=True)
    ap.add_argument("--gmc", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--nms", type=float, default=0.7)
    a = ap.parse_args()
    dev = torch.device("cuda")
    exp, model = frozen_detector(a.exp_file, a.ckpt, dev)
    size = tuple(exp.input_size)
    fusion = None
    if not a.no_fusion:
        state = torch.load(a.fusion, map_location="cpu", weights_only=False)
        fusion = TemporalFusion(state["channels"], state["align"]).to(dev).eval()
        fusion.load_state_dict(state["state_dict"])
    a.output.mkdir(parents=True, exist_ok=False)
    summary = {"detector_ckpt_sha256": sha256(a.ckpt), "fusion": str(a.fusion) if fusion else None,
               "fusion_sha256": sha256(a.fusion) if fusion else None, "conf": a.conf, "nms": a.nms, "size": list(size), "sequences": []}
    for s in a.sequences:
        length, w, h, img_dir, ext = seq_info(a.view / s)
        warps = load_gmc(a.gmc / f"{s}.txt")
        rows, prev = [], None
        for t in range(1, length + 1):
            img = cv2.imread(str(a.view / s / img_dir / f"{t:06d}{ext}"))
            x, r = preproc(img, size, MEAN, STD)
            with torch.no_grad():
                feats = model.backbone(torch.from_numpy(x).unsqueeze(0).to(dev))
                use = list(feats)
                if fusion is not None and prev is not None:
                    sc = np.diag([r, r, 1.0])
                    m_in = sc @ warps.get(t, np.eye(3)) @ np.linalg.inv(sc)
                    m_inv = torch.from_numpy(np.linalg.inv(m_in).astype(np.float32)).unsqueeze(0).to(dev)
                    use = fusion(list(feats), list(prev), m_inv, size)
                out = postprocess(model.head(use), exp.num_classes, a.conf, a.nms)[0]
            prev = feats
            if out is None:
                continue
            o = out.cpu().numpy()
            b = o[:, :4] / r
            for (x1, y1, x2, y2), sc_ in zip(b, o[:, 4] * o[:, 5]):
                rows.append(f"{t},-1,{x1:.4f},{y1:.4f},{x2 - x1:.4f},{y2 - y1:.4f},{sc_:.6f},-1,-1,-1\n")
        (a.output / s).mkdir()
        path = a.output / s / "det.txt"
        path.write_text("".join(rows))
        summary["sequences"].append({"sequence": s, "frames": length, "detections": len(rows), "det_sha256": sha256(path)})
        print(json.dumps(summary["sequences"][-1]), flush=True)
    (a.output / "export_summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
