"""Fixed-cache joint association inspired by TrackTrack TPA (CVPR 2025).

Source algorithm: kamkyu94/TrackTrack ee7f1c5fcbdcac48ed8bfab38d52c0006bf304da,
3. Tracker/trackers/utils.py::associate, iterative_assignment. This independently
implemented adapter retains ByteTrack high-score fused IoU costs and low-score
raw IoU costs, with a low-score penalty. No ReID, HMIoU, angle/confidence
prediction, second NMS cache, TAI, or altered Kalman/lifecycle/output policy.
"""
import numpy as np
from . import matching


def iterative_assignment(cost, threshold, reduce_step=0.05):
    """Mutual row/column minima; strict threshold, first-index tie breaking.

    Matched rows/columns are set to one as in the source. Input must be finite
    and clipped to [0,1], and threshold <= 1 keeps masked entries unmatchable.
    """
    cost = np.asarray(cost, dtype=float).copy()
    if cost.ndim != 2 or not np.isfinite(cost).all() or np.any((cost < 0) | (cost > 1)):
        raise ValueError("expected finite [0,1] cost matrix")
    if not 0 < threshold <= 1 or not 0 <= reduce_step <= 1:
        raise ValueError("invalid threshold or reduction")
    pairs = []
    if cost.shape[0] and cost.shape[1]:
        while True:
            row_best, col_best = cost.argmin(1), cost.argmin(0)
            found = [(i, int(j)) for i, j in enumerate(row_best)
                     if col_best[j] == i and cost[i, j] < threshold]
            threshold -= reduce_step
            if not found:
                break
            pairs.extend(found)
            for i, j in found:
                cost[i, :], cost[:, j] = 1., 1.
    used_t, used_d = {i for i, _ in pairs}, {j for _, j in pairs}
    return (np.asarray(pairs, dtype=int).reshape(-1, 2),
            tuple(i for i in range(cost.shape[0]) if i not in used_t),
            tuple(j for j in range(cost.shape[1]) if j not in used_d))


def assign(tracks, high, low, args, spec):
    """Both tracked and lost tracks compete for the same high+low pool."""
    hi = matching.iou_distance(tracks, high)
    if not args.mot20:
        hi = matching.fuse_score(hi, high)
    lo = matching.iou_distance(tracks, low) + spec["low_penalty"]
    cost = np.clip(np.concatenate([hi, lo], axis=1), 0., 1.)
    if spec["mode"] == "tracktrack":
        return iterative_assignment(cost, args.match_thresh, spec["reduce_step"])
    if spec["mode"] == "hungarian":
        # Match the strict TPA cost boundary; LAPJV uses an inclusive cost limit.
        return matching.linear_assignment(cost, thresh=np.nextafter(float(args.match_thresh), -np.inf))
    raise ValueError("unknown joint association mode")
