"""Run FCTrack arms on frozen detection caches and evaluate them with the pinned evaluator.

The association is the unmodified ByteTrack of this checkout and always consumes the detector's boxes and scores.
An arm without `recover` and `outlier` is the default ByteTrack. Ground truth is read only by the evaluator.
"""
import argparse
import configparser
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from fctrack.tracker import FCTrack, wrap_stracks  # noqa: E402
from homa.evaluate import evaluate  # noqa: E402
from yolox.tracker.basetrack import BaseTrack  # noqa: E402
from yolox.tracker.byte_tracker import BYTETracker, STrack  # noqa: E402

KEYS = ("MOTA", "IDF1", "HOTA", "DetA", "AssA", "FP", "FN", "IDSW")


def seq_info(seq_dir):
    ini = configparser.ConfigParser()
    ini.read(seq_dir / "seqinfo.ini")
    return {k: int(ini["Sequence"][k]) for k in ("seqLength", "imHeight", "imWidth", "frameRate")}


ATTEMPT_COLUMNS = ("frame,track_id,gap,since_real,obs,estimator,prop_x,prop_y,prop_w,prop_h,"
                   "final_x,final_y,final_w,final_h,p,decision,cov_x,cov_y")


def run_arm(root, cfg, cell, arm, tracks, device, limit, attempts_dir=None):
    wrap_stracks(STrack)
    verify = None
    if arm.get("verifier"):
        import torch
        from fctrack.verifier import load, verify as apply
        spec = cell["verifiers"][arm["verifier"]]
        model = load(spec["kind"], root / spec["weights"], torch.device(device))
        verify = lambda img, boxes: apply(spec["kind"], model, img, boxes, torch.device(device))
    data = root / cell["dataset_root"] / cell["split"]
    names = sorted(x.name for x in (root / cell["detections"]).glob("UAVSwarm-*") if x.is_dir())
    names = names[:limit] if limit else names
    tracks.mkdir(parents=True, exist_ok=False)
    if attempts_dir is not None:
        attempts_dir.mkdir(parents=True, exist_ok=False)
    stats, digests = {}, {}
    for name in names:
        info = seq_info(data / name)
        shape = (info["imHeight"], info["imWidth"])
        raw = np.loadtxt(root / cell["detections"] / name / "det.txt", delimiter=",", ndmin=2)
        targs = {**cfg["tracker"], **arm.get("tracker", {})}  # per-arm overrides (diagnostics; Exp074 track_buffer / freeze_lost)
        if arm.get("gmc"):  # Exp074: frame-to-frame camera motion estimated beforehand (Exp055 estimator), row `f a b tx c d ty`
            w = np.loadtxt(root / cell["gmc"] / f"{name}.txt", ndmin=2)
            targs["gmc_warps"] = {int(r[0]): r[1:].reshape(2, 3) for r in w}
        tracker = BYTETracker(SimpleNamespace(**targs), frame_rate=info["frameRate"])
        BaseTrack._count = 0
        fc = FCTrack(tracker, STrack, arm, verify, log=attempts_dir is not None)
        rows = []
        for frame in range(1, info["seqLength"] + 1):
            r = raw[raw[:, 0] == frame]
            det = np.column_stack((r[:, 2:4], r[:, 2:4] + r[:, 4:6], r[:, 6])).astype(np.float32)
            cache = []

            def image():
                if not cache:
                    cache.append(cv2.imread(str(data / name / "img1" / f"{frame:06d}.jpg")))
                return cache[0]

            for tid, box, score in fc.step(det, shape, image):
                x, y, w, h = box
                if w * h > 0:
                    rows.append("{},{},{},{},{},{},{},-1,-1,-1\n".format(frame, tid, round(x, 1), round(y, 1), round(w, 1), round(h, 1), round(score, 2)))
        text = "".join(rows)
        (tracks / f"{name}.txt").write_text(text)
        digests[name] = hashlib.sha256(text.encode()).hexdigest()
        if attempts_dir is not None:  # one row per verified fill-in attempt (read only: the tracks above are unchanged)
            (attempts_dir / f"{name}.csv").write_text(ATTEMPT_COLUMNS + "\n" + "".join(
                ",".join(repr(v) if isinstance(v, float) else str(v) for v in a) + "\n" for a in fc.attempts))
        for k, v in fc.stats.items():
            stats[k] = stats.get(k, 0) + v
    return names, stats, digests


def boot(delta, seed, n):
    rng = np.random.default_rng(seed)
    s = delta[rng.integers(0, len(delta), (n, len(delta)))].sum(1)
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="research repository root (datasets and experiments live under it)")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="smoke test only: the first k sequences")
    ap.add_argument("--log-attempts", action="store_true", help="write one csv row per verified fill-in attempt to <output>/attempts/<arm>/")
    a = ap.parse_args()
    root = a.root.resolve()
    cfg = json.loads(a.config.read_text())
    cell = cfg["cells"][a.cell]
    arms = {k: cfg["arms"][k] for k in (a.arms or cfg["arms"])}
    out = a.output.resolve()
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    res = {}
    for arm_name, arm in arms.items():
        tracks = out / "tracks" / arm_name
        names, stats, digests = run_arm(root, cfg, cell, arm, tracks, a.device, a.limit,
                                        out / "attempts" / arm_name if a.log_attempts else None)
        e = evaluate(root / cell["dataset_root"], cell["split"], tracks, out / "metrics" / f"{arm_name}.json", names)
        m, h = e["overall"]["motmetrics"], e["overall"]["hota"]
        overall = {"MOTA": m["mota"], "IDF1": m["idf1"], "HOTA": h["HOTA"], "DetA": h["DetA"], "AssA": h["AssA"],
                   "FP": m["num_false_positives"], "FN": m["num_misses"], "IDSW": m["num_switches"], "GT": m["num_objects"]}
        per = {n: {k: e["per_sequence"][n]["motmetrics"][k] for k in ("num_false_positives", "num_misses", "num_switches", "num_objects")} for n in names}
        res[arm_name] = {"overall": overall, "stats": stats, "per_sequence": per, "track_sha256": digests}
        same = None
        if arm_name == "base" and cell.get("reference_tracks") and not a.limit:
            same = sum((root / cell["reference_tracks"] / f"{n}.txt").read_bytes() == (tracks / f"{n}.txt").read_bytes() for n in names)
            res[arm_name]["identical_to_reference"] = [same, len(names)]
        print(a.cell, arm_name, json.dumps({k: round(overall[k], 3) if isinstance(overall[k], float) else overall[k] for k in KEYS}), json.dumps(stats),
              "" if same is None else f"identical to reference {same}/{len(names)}", flush=True)
    if "base" in res:
        b = res["base"]
        err = lambda r, n: r["per_sequence"][n]["num_false_positives"] + r["per_sequence"][n]["num_misses"] + r["per_sequence"][n]["num_switches"]
        gt = b["overall"]["GT"]
        for arm_name, r in res.items():
            if arm_name == "base":
                continue
            names = sorted(b["per_sequence"])
            d = np.array([err(b, n) - err(r, n) for n in names], float)  # fewer errors than the baseline = positive
            r["vs_base"] = {k: r["overall"][k] - b["overall"][k] for k in KEYS}
            r["vs_base"]["MOTA_ci95"] = [x / gt * 100 for x in boot(d, cfg["bootstrap"]["seed"], cfg["bootstrap"]["resamples"])]
            r["vs_base"]["sequences_better_worse"] = [int((d > 0).sum()), int((d < 0).sum())]
            print(a.cell, arm_name, "vs base", json.dumps({k: round(v, 3) for k, v in r["vs_base"].items() if not isinstance(v, list)}),
                  [round(x, 2) for x in r["vs_base"]["MOTA_ci95"]], r["vs_base"]["sequences_better_worse"], flush=True)
    (out / "metrics.json").write_text(json.dumps({"cell": a.cell, "arms": res}, indent=1) + "\n")


if __name__ == "__main__":
    main()
