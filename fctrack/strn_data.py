"""Exp086: turn the logged lost-track states of one tracking run into STRN training samples (reads annotations).

For every state: the model inputs of fctrack.strn.inputs, and the labels
  exist  the member this track followed is annotated in the frame, its centre is inside the search window, and no
         tracked box covers it (IoU >= 0.5);
  box_t  that annotation relative to the proposal (dx, dy, log w, log h in proposal sides);
  ptr_t  the candidate box with the largest IoU >= 0.5 with that annotation, else N_CAND (use the regressed box).
The followed member is the annotated id matched (IoU >= 0.5) most often by the track's own written rows up to the
frame it was lost. A track that never matched an annotation has only negative states.
Usage: python -m fctrack.strn_data <run output dir> <arm> <dataset root> <split> <detections dir> <out.npz>
"""
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from fctrack.strn import N_CAND, SEARCH_ZOOM, _iou, inputs


def main():
    out_dir, arm, root, split, dets, dst = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5]), Path(sys.argv[6])
    out_dir = Path(out_dir)
    keys = ("srch", "tmpl", "cand", "cmask", "cbox", "nb", "nmask", "state", "side", "centre")
    acc = {k: [] for k in keys + ("exist", "box_t", "ptr_t", "seq", "frame", "track")}
    names = []
    for sf in sorted((out_dir / "states" / arm).glob("*.json")):
        name = sf.stem
        names.append(name)
        log = json.loads(sf.read_text())
        if not log["states"]:
            continue
        gt = np.loadtxt(root / split / name / "gt/gt.txt", delimiter=",", ndmin=2)
        gtf = {int(k): gt[gt[:, 0] == k] for k in np.unique(gt[:, 0])}
        trk = out_dir / "tracks" / arm / f"{name}.txt"
        rows = np.loadtxt(trk, delimiter=",", ndmin=2) if trk.stat().st_size else np.zeros((0, 10))
        follow = {}  # track id -> list of (frame, matched gt id)
        for r in rows:
            g = gtf.get(int(r[0]))
            if g is None:
                continue
            k = _iou(r[2:6], g[:, 2:6])[0]
            if k.max() >= 0.5:
                follow.setdefault(int(r[1]), []).append((int(r[0]), int(g[int(k.argmax()), 1])))
        raw = np.loadtxt(dets / name / "det.txt", delimiter=",", ndmin=2)
        img_cache = {}

        def img(k):
            if k not in img_cache:
                if len(img_cache) > 64:
                    img_cache.pop(next(iter(img_cache)))
                img_cache[k] = cv2.imread(str(root / split / name / "img1" / f"{k:06d}.jpg"))
            return img_cache[k]

        for s in log["states"]:
            f = s["frame"]
            r = raw[raw[:, 0] == f]
            det = np.column_stack((r[:, 2:4], r[:, 2:4] + r[:, 4:6], r[:, 6])).astype(np.float32)
            tracked = np.asarray(log["tracked"].get(str(f), []), float).reshape(-1, 4)
            x = inputs(s, img(f), img(max(f - 1, 1)), img(s["rframe"]), det, tracked)
            m = [gid for fr, gid in follow.get(s["track_id"], []) if fr <= s["lost_frame"]]
            exist, box_t, ptr_t = 0.0, np.zeros(4, np.float32), N_CAND
            if m:
                gid = Counter(m).most_common(1)[0][0]
                g = gtf.get(f)
                g = g[g[:, 1] == gid] if g is not None else np.zeros((0, 10))
                if len(g):
                    gb = g[0, 2:6].astype(float)
                    c = gb[:2] + gb[2:] / 2
                    half = SEARCH_ZOOM / 2 * x["side"]
                    inside = abs(c[0] - x["centre"][0]) <= half and abs(c[1] - x["centre"][1]) <= half
                    covered = len(tracked) and _iou(gb, tracked).max() >= 0.5
                    if inside and not covered:
                        exist = 1.0
                        sd = float(x["side"])
                        box_t = np.array([(c[0] - x["centre"][0]) / sd, (c[1] - x["centre"][1]) / sd,
                                          np.log(max(gb[2], 1) / sd), np.log(max(gb[3], 1) / sd)], np.float32)
                        if x["cmask"].any():
                            ov = _iou(gb, x["cbox"])[0] * x["cmask"]
                            if ov.max() >= 0.5:
                                ptr_t = int(ov.argmax())
            for k in keys:
                acc[k].append(x[k])
            acc["exist"].append(np.float32(exist)); acc["box_t"].append(box_t); acc["ptr_t"].append(np.int64(ptr_t))
            acc["seq"].append(name); acc["frame"].append(f); acc["track"].append(s["track_id"])
    arrays = {k: np.stack(v) if v else np.zeros(0) for k, v in acc.items()}
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **arrays)
    n = len(acc["exist"])
    print(json.dumps({"sequences": len(names), "states": n, "positives": int(sum(acc["exist"])),
                      "pointer_to_candidate": int(sum(1 for e, p in zip(acc["exist"], acc["ptr_t"]) if e and p < N_CAND))}))


if __name__ == "__main__":
    main()
