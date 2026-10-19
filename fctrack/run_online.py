"""Exp116: run the FCTrack arms with the detector inside the loop (fctrack.online_det.OnlineDetector).

An arm is a tracker arm of the usual configuration plus an `online` block (adaptation parameters of the detector's objectness
layers; `lr` 0 = the source detector, whose detections equal the cached ones). One process = one arm = one detector pass per
sequence. Ground truth is read only by the evaluator.

Usage: run_online.py --root . --config <cfg.json> --cell <cell> --arm <arm> --output <dir> [--limit k] [--dump-det]
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from fctrack.online_det import OnlineDetector  # noqa: E402
from fctrack.run import KEYS, seq_info  # noqa: E402
from fctrack.tracker import FCTrack, wrap_stracks  # noqa: E402
from homa.evaluate import evaluate  # noqa: E402
from yolox.tracker.basetrack import BaseTrack  # noqa: E402
from yolox.tracker.byte_tracker import BYTETracker, STrack  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump-det", action="store_true")
    a = ap.parse_args()
    root = a.root.resolve()
    cfg = json.loads(a.config.read_text())
    cell, arm = cfg["cells"][a.cell], cfg["arms"][a.arm]
    dcfg = cfg["detector"]
    dev = torch.device(a.device)
    wrap_stracks(STrack)
    verify = None
    if arm.get("verifier"):
        from fctrack.verifier import load, verify as apply
        spec = cell["verifiers"][arm["verifier"]]
        vm = load(spec["kind"], root / spec["weights"], dev)
        verify = lambda img, boxes: apply(spec["kind"], vm, img, boxes, dev)
    det = OnlineDetector(root / dcfg["exp_file"], root / dcfg["checkpoint"], dev, adapt=arm.get("online", {"lr": 0.0}))
    data = root / cell["dataset_root"] / cell["split"]
    names = sorted(x.name for x in (data).glob("UAVSwarm-*") if x.is_dir())
    if cell.get("sequences"):  # optional subset of the split (e.g. the verifier's held-out sequences)
        names = [n for n in names if n in set(cell["sequences"])]
    names = names[:a.limit] if a.limit else names
    tracks = a.output / "tracks" / a.arm
    tracks.mkdir(parents=True, exist_ok=False)
    stats, t0 = {}, time.time()
    for name in names:
        info = seq_info(data / name)
        shape = (info["imHeight"], info["imWidth"])
        tracker = BYTETracker(SimpleNamespace(**cfg["tracker"]), frame_rate=info["frameRate"])
        BaseTrack._count = 0
        fc = FCTrack(tracker, STrack, arm, verify, log=False)
        det.reset()
        det_rows, rows_out, age = [], [], {}
        for frame in range(1, info["seqLength"] + 1):
            img = cv2.imread(str(data / name / "img1" / f"{frame:06d}.jpg"))
            d = det.detect(img)
            if a.dump_det:
                det_rows += ["{},-1,{:.4f},{:.4f},{:.4f},{:.4f},{:.6f},-1,-1,-1\n".format(frame, x1, y1, x2 - x1, y2 - y1, s) for x1, y1, x2, y2, s in d]
            written = []
            for tid, box, score in fc.step(d, shape, lambda: img):
                x, y, w, h = box
                if w * h > 0:
                    rows_out.append("{},{},{},{},{},{},{},-1,-1,-1\n".format(frame, tid, round(x, 1), round(y, 1), round(w, 1), round(h, 1), round(score, 2)))
                    age[tid] = age.get(tid, 0) + 1
                    written.append((box, age[tid]))
            det.adapt(written, d)
        (tracks / f"{name}.txt").write_text("".join(rows_out))
        if a.dump_det:
            (a.output / "det" / a.arm / name).mkdir(parents=True, exist_ok=True)
            (a.output / "det" / a.arm / name / "det.txt").write_text("".join(det_rows))
        for k, v in list(fc.stats.items()) + list(det.stats.items()):
            stats[k] = stats.get(k, 0) + v
        print(name, f"{time.time() - t0:.0f}s", flush=True)
    e = evaluate(root / cell["dataset_root"], cell["split"], tracks, a.output / "metrics" / f"{a.arm}.json", names) if (a.output / "metrics").mkdir(parents=True, exist_ok=True) is None else None
    m, h = e["overall"]["motmetrics"], e["overall"]["hota"]
    overall = {"MOTA": m["mota"], "IDF1": m["idf1"], "HOTA": h["HOTA"], "DetA": h["DetA"], "AssA": h["AssA"], "FP": m["num_false_positives"], "FN": m["num_misses"],
               "IDSW": m["num_switches"], "GT": m["num_objects"]}
    per = {n: {k: e["per_sequence"][n]["motmetrics"][k] for k in ("num_false_positives", "num_misses", "num_switches", "num_objects")} for n in names}
    (a.output / "metrics.json").write_text(json.dumps({"cell": a.cell, "arms": {a.arm: {"overall": overall, "stats": stats, "per_sequence": per}}, "seconds": round(time.time() - t0, 1)}, indent=1) + "\n")
    print(a.cell, a.arm, json.dumps({k: round(overall[k], 3) if isinstance(overall[k], float) else overall[k] for k in KEYS}), json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
