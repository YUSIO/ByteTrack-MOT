"""Exp076: apply the online birth relabelling (fctrack/relink.py) to saved tracker outputs over a grid of
(family, tau, gap) and evaluate every setting with the pinned evaluator. Ground truth is read only by the evaluator.

Writes <out>/tracks/<setting>/<sequence>.txt, <out>/metrics/<setting>.json, <out>/relinks/<setting>.csv and
<out>/grid.json (overall metrics plus the per-sequence counts needed for a sequence bootstrap of dIDF1).
--subset NAME=seq,seq,... adds a second evaluation over a subset of sequences (reported under that name).
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from fctrack.relink import FAMILIES, relabel  # noqa: E402
from homa.evaluate import evaluate  # noqa: E402


def load(p):
    return np.loadtxt(p, delimiter=",", ndmin=2) if p.stat().st_size else np.zeros((0, 10))


def counts(e, names):
    per = {}
    for n in names:
        m = e["per_sequence"][n]["motmetrics"]
        idtp = m["idr"] / 100.0 * m["num_objects"]
        per[n] = {"idtp": idtp, "gt": m["num_objects"], "pred": m["num_predictions"], "fp": m["num_false_positives"],
                  "fn": m["num_misses"], "idsw": m["num_switches"]}
    return per


def one(job):
    a, setting, family, tau, gap = job
    names = sorted(p.stem for p in a["tracks"].glob("*.txt"))
    d = a["out"] / "tracks" / setting
    d.mkdir(parents=True, exist_ok=False)
    rel = []
    for n in names:
        rows = load(a["tracks"] / f"{n}.txt")
        if family != "none":
            warps = {}
            if family == "gmc":
                w = np.loadtxt(a["warps"] / f"{n}.txt", ndmin=2)
                warps = {int(r[0]): r[1:].reshape(2, 3) for r in w}
            log = []
            rows = relabel(rows, family, tau, gap, warps, log)
            rel += [(n, *x) for x in log]
        (d / f"{n}.txt").write_text("".join("{},{},{},{},{},{},{},-1,-1,-1\n".format(int(r[0]), int(r[1]), r[2], r[3], r[4], r[5], r[6]) for r in rows))
    (a["out"] / "relinks" / f"{setting}.csv").write_text("sequence,frame,new_id,label,gap,distance\n" + "".join(
        f"{n},{f},{t},{L},{g},{dist:.4f}\n" for n, f, t, L, g, dist in rel))
    res = {"family": family, "tau": tau, "gap": gap, "relinks": len(rel)}
    for name, subset in [("all", names)] + [(k, [x for x in names if x in v]) for k, v in a["subsets"].items()]:
        e = evaluate(a["data"], a["split"], d, a["out"] / "metrics" / f"{setting}.{name}.json", subset)
        m, h = e["overall"]["motmetrics"], e["overall"]["hota"]
        res[name] = {"IDF1": m["idf1"], "MOTA": m["mota"], "HOTA": h["HOTA"], "AssA": h["AssA"], "IDSW": m["num_switches"],
                     "FP": m["num_false_positives"], "FN": m["num_misses"], "per_sequence": counts(e, subset)}
    return setting, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--warps", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--taus", nargs="+", type=float, required=True)
    ap.add_argument("--gaps", nargs="+", type=int, required=True)
    ap.add_argument("--settings", nargs="*", help="explicit family:tau:gap settings instead of the grid")
    ap.add_argument("--subset", action="append", default=[])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "metrics").mkdir(); (a.out / "relinks").mkdir()
    cfg = {"data": a.data, "split": a.split, "tracks": a.tracks, "warps": a.warps, "out": a.out,
           "subsets": {s.split("=")[0]: s.split("=")[1].split(",") for s in a.subset}}
    jobs = [(cfg, "none", "none", 0.0, 0)]
    if a.settings:
        for s in a.settings:
            f, t, g = s.split(":")
            jobs.append((cfg, f"{f}.t{float(t):g}.g{int(g)}", f, float(t), int(g)))
    else:
        jobs += [(cfg, f"{f}.t{t:g}.g{g}", f, t, g) for f in a.families for t in a.taus for g in a.gaps]
    grid = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for setting, res in ex.map(one, jobs):
            grid[setting] = res
            print(setting, res["relinks"], json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res["all"].items() if k != "per_sequence"}), flush=True)
    (a.out / "grid.json").write_text(json.dumps(grid, indent=1) + "\n")


if __name__ == "__main__":
    main()
