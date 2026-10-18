"""Write a re-scored detection cache: the score of every row that is in the candidate table is replaced by the network's
probability; rows below the table's `min_score` keep their score. Boxes, frames and file format are unchanged.
  apply.py --dets <cache root> --pred <oof.npz | applied.npz> --out <cache root> [--mix w]   (score = w * p + (1 - w) * score)
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mix", type=float, default=1.0)
    ap.add_argument("--min-score", type=float, default=0.1)
    ap.add_argument("--band", type=float, nargs=2, default=(0.0, 1.01), help="only rows whose original score is in [lo, hi) are replaced")
    a = ap.parse_args()
    P = np.load(a.pred)
    seq, frame, box, p = P["seq"], P["frame"], P["box"], P["p"]
    a.out.mkdir(parents=True, exist_ok=False)
    for sd in sorted(a.dets.glob("UAVSwarm-*")):
        n = int(sd.name[-2:])
        raw = np.loadtxt(sd / "det.txt", delimiter=",", ndmin=2)
        m = seq == n
        key = {(int(f), round(float(b[0]), 3), round(float(b[1]), 3)): float(q) for f, b, q in zip(frame[m], box[m], p[m])}
        rows = []
        for r in raw:
            q = key.get((int(r[0]), round(float(r[2]), 3), round(float(r[3]), 3)))
            sc = r[6] if (q is None or not a.band[0] <= r[6] < a.band[1]) else a.mix * q + (1 - a.mix) * r[6]
            rows.append(f"{int(r[0])},-1,{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f},{sc:.6f},-1,-1,-1\n")
        (a.out / sd.name).mkdir()
        (a.out / sd.name / "det.txt").write_text("".join(rows))
        print(sd.name, len(rows), int(m.sum()), flush=True)


if __name__ == "__main__":
    main()
