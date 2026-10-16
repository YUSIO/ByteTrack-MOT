"""Exp082: structure terms in ByteTrack's association cost (off unless `args.struct` is set).

The IoU cost c_iou of a stage (1 - IoU, fused with the detection score in stages 1 and 3) is blended with a structure
cost c_s in [0, 1]:   cost = (1 - w) * c_iou + w * c_s.   Thresholds, stages and lifecycle are unchanged.

Frame context (built once per frame after the Kalman prediction and the optional camera-motion warp):
  reference tracks   the activated Tracked tracks (predicted centres)
  reference boxes    every detector box with score > 0.1 (centres)
Terms, with s measured in units of the track's predicted box side sqrt(w h):
  cen      position relative to the swarm centroid must agree:
             s = | (p_track - mean(reference tracks)) - (p_det - mean(reference boxes)) | / side,  c = 1 - exp(-s / 2)
  cenmed   the same with the coordinate-wise median instead of the mean
  knn      the K nearest other reference objects must be at the same distances (sorted distance vectors, K = 3):
             s = mean_k | d_track,k - d_det,k | / side,  c = 1 - exp(-s)
  cen+knn  c = (c_cen + c_knn) / 2
  knnshuf  control: knn with the detections' distance vectors permuted among the reference boxes of the frame
Where a term is undefined (fewer than 2 reference tracks or boxes, no neighbours) the pair keeps c_iou.
"""
import numpy as np

K = 3


def _centre(tlwh):
    return np.array([tlwh[0] + tlwh[2] / 2.0, tlwh[1] + tlwh[3] / 2.0])


class FrameContext:
    def __init__(self, cfg, ref_tracks, ref_box_centres, frame_id):
        self.cfg = cfg
        self.ref_tracks = ref_tracks
        self.tc = np.array([_centre(t.tlwh) for t in ref_tracks]).reshape(-1, 2)
        self.dc = np.asarray(ref_box_centres, float).reshape(-1, 2)
        self.active = len(self.tc) >= 2 and len(self.dc) >= 2
        mode = cfg["mode"]
        self.mode = mode
        if not self.active:
            return
        if mode in ("cen", "cenmed", "cen+knn"):
            f = np.median if mode == "cenmed" else np.mean
            self.Ct, self.Cd = f(self.tc, axis=0), f(self.dc, axis=0)
        if mode in ("knn", "cen+knn", "knnshuf"):
            D = np.linalg.norm(self.dc[:, None] - self.dc[None], axis=2)
            np.fill_diagonal(D, np.inf)
            Ds = np.sort(D, axis=1)[:, :K]
            self.dknn = np.where(np.isfinite(Ds), Ds, np.nan)
            if mode == "knnshuf":
                self.dknn = self.dknn[np.random.RandomState(frame_id).permutation(len(self.dknn))]
            if self.dknn.shape[1] < K:
                self.dknn = np.pad(self.dknn, ((0, 0), (0, K - self.dknn.shape[1])), constant_values=np.nan)

    def track_knn(self, tracks):
        out = np.full((len(tracks), K), np.nan)
        for i, t in enumerate(tracks):
            d = np.linalg.norm(self.tc - _centre(t.tlwh), axis=1)
            d = np.sort(np.array([x for x, r in zip(d, self.ref_tracks) if r is not t]))[:K]
            out[i, :len(d)] = d
        return out

    def blend(self, c_iou, tracks, dets):
        if not self.active or c_iou.size == 0:
            return c_iou
        w = self.cfg["w"]
        pc = np.array([_centre(t.tlwh) for t in tracks])
        side = np.sqrt(np.maximum(np.array([t.tlwh[2] * t.tlwh[3] for t in tracks]), 1.0))[:, None]
        terms = []
        if self.mode in ("cen", "cenmed", "cen+knn"):
            dp = np.array([_centre(d.tlwh) for d in dets])
            s = np.linalg.norm((pc - self.Ct)[:, None] - (dp - self.Cd)[None], axis=2) / side
            terms.append(1.0 - np.exp(-s / 2.0))
        if self.mode in ("knn", "cen+knn", "knnshuf"):
            dt = self.track_knn(tracks)
            dd = self.dknn[[d.ridx for d in dets]]
            diff = np.abs(dt[:, None, :] - dd[None, :, :])
            with np.errstate(all="ignore"):
                s = np.nanmean(diff, axis=2) / side
            terms.append(np.where(np.isfinite(s), 1.0 - np.exp(-s), np.nan))
        c_s = terms[0] if len(terms) == 1 else (terms[0] + terms[1]) / 2.0
        c_s = np.where(np.isfinite(c_s), c_s, c_iou)
        return (1.0 - w) * c_iou + w * c_s
