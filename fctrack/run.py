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

from fctrack.tracker import FCTrack, iou_matrix, wrap_stracks  # noqa: E402
from homa.evaluate import evaluate  # noqa: E402
from yolox.tracker.basetrack import BaseTrack  # noqa: E402
from yolox.tracker.byte_tracker import BYTETracker, STrack  # noqa: E402

KEYS = ("MOTA", "IDF1", "HOTA", "DetA", "AssA", "FP", "FN", "IDSW")


def seq_info(seq_dir):
    ini = configparser.ConfigParser()
    ini.read(seq_dir / "seqinfo.ini")
    return {k: int(ini["Sequence"][k]) for k in ("seqLength", "imHeight", "imWidth", "frameRate")}


def slot_guarded(det, prior, slots, cfg):
    """Exp093: merge the boxes of a track-conditioned detector pass (`prior`) into the plain pass (`det`) at the
    tracker's member slots only, one box per slot, never lowering anything. Both are n x 5 (x1, y1, x2, y2, score);
    slots are k x 4 (tlwh) one-step predictions of the tracked and lost tracks.

    A slot that already has a plain box scoring >= `strong` within `radius_sides` slot sides of its centre is explained
    and claims nothing. Every other slot claims the highest-scoring prior box (score >= `min_score`) within that radius
    that no other slot has claimed (slots in order of their best candidate's score). The claimed box raises the score of
    the plain box it overlaps most (IoU >= `merge_iou`; geometry kept) or, when no plain box overlaps it by
    `free_iou` or more, is added. Boxes no slot claims are left exactly as the plain pass gave them."""
    if len(prior) == 0 or len(slots) == 0:
        return det, 0, 0
    sc = slots[:, :2] + slots[:, 2:] / 2
    side = np.sqrt(np.maximum(slots[:, 2] * slots[:, 3], 1.0))
    centre = lambda b: (b[:, :2] + b[:, 2:4]) / 2
    within = lambda b: np.linalg.norm(centre(b)[None] - sc[:, None], axis=2) <= cfg["radius_sides"] * side[:, None]  # k x n
    strong = det[det[:, 4] >= cfg["strong"]] if len(det) else det
    open_ = ~within(strong).any(1) if len(strong) else np.ones(len(slots), bool)
    prior = prior[prior[:, 4] >= cfg["min_score"]]
    if len(prior) == 0 or not open_.any():
        return det, 0, 0
    cand = within(prior) & open_[:, None]
    det = det.copy()
    to_tlwh = lambda b: np.column_stack((b[:, :2], b[:, 2:4] - b[:, :2]))
    raised = added = 0
    used, new = set(), []
    pairs = sorted(((prior[j, 4], i, j) for i, j in zip(*np.nonzero(cand))), reverse=True)
    done = set()
    for _, i, j in pairs:
        if i in done or j in used:
            continue
        done.add(i); used.add(j)
        m = iou_matrix(to_tlwh(prior[j:j + 1]), to_tlwh(det))[0] if len(det) else np.zeros(0)
        if len(m) and m.max() >= cfg["merge_iou"]:
            k = int(m.argmax())
            if prior[j, 4] > det[k, 4]:
                det[k, 4] = prior[j, 4]
                raised += 1
        elif not len(m) or m.max() < cfg["free_iou"]:
            new.append(prior[j])
            added += 1
    if new:
        det = np.vstack((det, np.array(new, det.dtype)))
    return det, raised, added


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
        fus = arm.get("fusion")  # Exp093: a second, track-conditioned detection cache merged only at member slots
        if fus:
            praw = np.loadtxt(root / cell["fusion_caches"][fus["source"]] / name / "det.txt", delimiter=",", ndmin=2)
        targs = {**cfg["tracker"], **arm.get("tracker", {})}  # per-arm overrides (diagnostics; Exp074 track_buffer / freeze_lost)
        if arm.get("gmc"):  # Exp074: frame-to-frame camera motion estimated beforehand (Exp055 estimator), row `f a b tx c d ty`
            w = np.loadtxt(root / cell["gmc"] / f"{name}.txt", ndmin=2)
            targs["gmc_warps"] = {int(r[0]): r[1:].reshape(2, 3) for r in w}
        tracker = BYTETracker(SimpleNamespace(**targs), frame_rate=info["frameRate"])
        BaseTrack._count = 0
        fc = FCTrack(tracker, STrack, arm, verify, log=attempts_dir is not None)
        fstats = fc.stats
        if arm.get("relabel", {}).get("family") == "gmc":  # Exp077: camera motion for the gmc relabel family
            w = np.loadtxt(root / cell["gmc"] / f"{name}.txt", ndmin=2)
            fc.warps = {int(r[0]): r[1:].reshape(2, 3) for r in w}
        rows = []
        for frame in range(1, info["seqLength"] + 1):
            r = raw[raw[:, 0] == frame]
            det = np.column_stack((r[:, 2:4], r[:, 2:4] + r[:, 4:6], r[:, 6])).astype(np.float32)
            if fus:
                q = praw[praw[:, 0] == frame]
                prior = np.column_stack((q[:, 2:4], q[:, 2:4] + q[:, 4:6], q[:, 6])).astype(np.float32)
                pool = [t for t in tracker.tracked_stracks if t.is_activated] + list(tracker.lost_stracks)
                slots = np.array([np.r_[t.mean[:2] + t.mean[4:6] - np.r_[t.mean[2] * t.mean[3], t.mean[3]] / 2, t.mean[2] * t.mean[3], t.mean[3]]
                                  for t in pool], float).reshape(-1, 4)  # one-step constant-velocity prediction, tlwh
                det, nr, na = slot_guarded(det, prior, slots, fus)
                fstats["raised"] += nr
                fstats["added"] += na
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
