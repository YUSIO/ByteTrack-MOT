"""Closed-loop arms of Exp071 on a frozen detection cache, evaluated with the pinned evaluator.

All arms of a cell advance through a sequence frame by frame together, so the detector features of a frame are
computed once (fp16 autocast, as in training) and shared. Each arm has its own BYTETracker and its own track-id
counter, so the default ByteTrack arm writes exactly what a run of its own writes. Arm kinds:

  {}                                  default ByteTrack
  {"verifier": ..., "recover": ...}   Exp070 FCTrack fill-in with a separate verification network on image crops
  {"head": ..., "recover": ...}       the same fill-in with a joint track head on the detector's features

Ground truth is read only by the evaluator.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from fctrack.run import KEYS, boot, seq_info  # noqa: E402
from fctrack.tracker import FCTrack, wrap_stracks  # noqa: E402
from homa.evaluate import evaluate  # noqa: E402
from jdt.detector import features, letterbox, load_detector, normalise  # noqa: E402
from jdt.head import TrackHead  # noqa: E402
from jdt.tracker import JDTTrack  # noqa: E402
from yolox.tracker.basetrack import BaseTrack  # noqa: E402
from yolox.tracker.byte_tracker import BYTETracker, STrack  # noqa: E402


def build(root, cfg, cell, arms, device):
    heads, verifiers = {}, {}
    for name, arm in arms.items():
        if arm.get("head") and arm["head"] not in heads:
            spec = cfg["heads"][arm["head"]]
            h = TrackHead(spec["groups"]).to(device)
            h.load_state_dict(torch.load(root / spec["weights"], map_location=device)[spec.get("key", arm["head"])])
            heads[arm["head"]] = h.eval()
        if arm.get("verifier") and arm["verifier"] not in verifiers:
            from fctrack.verifier import load, verify as apply
            spec = cfg["verifiers"][arm["verifier"]]
            model = load(spec["kind"], root / spec["weights"], device)
            verifiers[arm["verifier"]] = (lambda kind, model: (lambda img, boxes: apply(kind, model, img, boxes, device)))(spec["kind"], model)
    return heads, verifiers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    root = a.root.resolve()
    cfg = json.loads(a.config.read_text())
    cell = cfg["cells"][a.cell]
    arms = {k: cfg["arms"][k] for k in (a.arms or cfg["arms"])}
    out = a.output.resolve()
    (out / "metrics").mkdir(parents=True, exist_ok=False)
    device = torch.device(a.device)
    wrap_stracks(STrack)
    need_feat = any(arm.get("head") for arm in arms.values())
    det_model = load_detector(cfg["detector"]["exp_file"], root / cfg["detector"]["ckpt"], device)[1] if need_feat else None
    heads, verifiers = build(root, cfg, cell, arms, device)
    data = root / cell["dataset_root"] / cell["split"]
    names = sorted(x.name for x in (root / cell["detections"]).glob("UAVSwarm-*") if x.is_dir())
    names = names[: a.limit] if a.limit else names
    rows = {k: {} for k in arms}
    stats = {k: {} for k in arms}
    for name in names:
        info = seq_info(data / name)
        shape = (info["imHeight"], info["imWidth"])
        raw = np.loadtxt(root / cell["detections"] / name / "det.txt", delimiter=",", ndmin=2)
        runners, counters = {}, {}
        for k, arm in arms.items():
            BaseTrack._count = 0
            bt = BYTETracker(SimpleNamespace(**{**cfg["tracker"], **arm.get("tracker", {})}), frame_rate=info["frameRate"])
            if arm.get("head"):
                runners[k] = JDTTrack(bt, STrack, arm, heads[arm["head"]], device)
            else:
                runners[k] = FCTrack(bt, STrack, arm, verifiers.get(arm.get("verifier")))
            counters[k] = 0
            rows[k][name] = []
        for frame in range(1, info["seqLength"] + 1):
            r = raw[raw[:, 0] == frame] if len(raw) else raw
            det = np.column_stack((r[:, 2:4], r[:, 2:4] + r[:, 4:6], r[:, 6])).astype(np.float32) if len(r) else np.zeros((0, 5), np.float32)
            img = cv2.imread(str(data / name / "img1" / f"{frame:06d}.jpg"))
            if need_feat:
                lb, ratio = letterbox(img)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    p3, obj = features(det_model, normalise(torch.from_numpy(lb[None]).to(device)))
            for k, fc in runners.items():
                BaseTrack._count = counters[k]
                if isinstance(fc, JDTTrack):
                    fc.set_frame(p3, obj, ratio)
                for tid, box, score in fc.step(det, shape, lambda: img):
                    x, y, w, h = box
                    if w * h > 0:
                        rows[k][name].append("{},{},{},{},{},{},{},-1,-1,-1\n".format(frame, tid, round(x, 1), round(y, 1), round(w, 1), round(h, 1), round(score, 2)))
                counters[k] = BaseTrack._count
        for k, fc in runners.items():
            for s, v in fc.stats.items():
                stats[k][s] = stats[k].get(s, 0) + v
        print(name, "done", flush=True)
    res = {}
    for k in arms:
        tracks = out / "tracks" / k
        tracks.mkdir(parents=True)
        digests = {}
        for name in names:
            text = "".join(rows[k][name])
            (tracks / f"{name}.txt").write_text(text)
            digests[name] = hashlib.sha256(text.encode()).hexdigest()
        e = evaluate(root / cell["dataset_root"], cell["split"], tracks, out / "metrics" / f"{k}.json", names)
        m, h = e["overall"]["motmetrics"], e["overall"]["hota"]
        overall = {"MOTA": m["mota"], "IDF1": m["idf1"], "HOTA": h["HOTA"], "DetA": h["DetA"], "AssA": h["AssA"],
                   "FP": m["num_false_positives"], "FN": m["num_misses"], "IDSW": m["num_switches"], "GT": m["num_objects"]}
        per = {n: {x: e["per_sequence"][n]["motmetrics"][x] for x in ("num_false_positives", "num_misses", "num_switches", "num_objects")} for n in names}
        res[k] = {"overall": overall, "stats": stats[k], "per_sequence": per, "track_sha256": digests}
        same = None
        if k == "base" and cell.get("reference_tracks") and not a.limit:
            same = sum((root / cell["reference_tracks"] / f"{n}.txt").read_bytes() == (tracks / f"{n}.txt").read_bytes() for n in names)
            res[k]["identical_to_reference"] = [same, len(names)]
        print(a.cell, k, json.dumps({x: round(overall[x], 3) if isinstance(overall[x], float) else overall[x] for x in KEYS}), json.dumps(stats[k]),
              "" if same is None else f"identical to reference {same}/{len(names)}", flush=True)
    if "base" in res:
        b = res["base"]
        err = lambda r, n: r["per_sequence"][n]["num_false_positives"] + r["per_sequence"][n]["num_misses"] + r["per_sequence"][n]["num_switches"]
        gt = b["overall"]["GT"]
        for k, r in res.items():
            if k == "base":
                continue
            d = np.array([err(b, n) - err(r, n) for n in sorted(b["per_sequence"])], float)
            r["vs_base"] = {x: r["overall"][x] - b["overall"][x] for x in KEYS}
            r["vs_base"]["MOTA_ci95"] = [x / gt * 100 for x in boot(d, cfg["bootstrap"]["seed"], cfg["bootstrap"]["resamples"])]
            r["vs_base"]["sequences_better_worse"] = [int((d > 0).sum()), int((d < 0).sum())]
            print(a.cell, k, "vs base", json.dumps({x: round(v, 3) for x, v in r["vs_base"].items() if not isinstance(v, list)}),
                  [round(x, 2) for x in r["vs_base"]["MOTA_ci95"]], r["vs_base"]["sequences_better_worse"], flush=True)
    (out / "metrics.json").write_text(json.dumps({"cell": a.cell, "arms": res}, indent=1) + "\n")


if __name__ == "__main__":
    main()
