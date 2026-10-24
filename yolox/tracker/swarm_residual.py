"""Common innovation after KF + GMC, applied only to Lost states (Exp163).

Anchors are current high-score matches of previously Tracked members. Their
innovation is observation minus the already predicted and GMC-warped centre.
Neither camera displacement nor KF velocity is added twice. Correction after
all associations affects the next frame only. No separate shadow state can
bypass the next frame's GMC.
"""
import json
from collections import Counter

import numpy as np

from .basetrack import TrackState


class SwarmResidual:
    def __init__(self, width, height, mode="apply", min_anchors=2,
                 max_residual=0.04, ttl=30):
        self.width, self.height = float(width), float(height)
        self.diag = float(np.hypot(width, height))
        self.mode = mode
        self.min_anchors, self.max_residual, self.ttl = min_anchors, max_residual, ttl
        self.counts = Counter()
        self.shift = None
        self.shifts = []

    def collect(self, pool, detections, matches, frame_id):
        self.shift = None
        residuals = []
        for ti, di in matches:
            track = pool[ti]
            if track.state != TrackState.Tracked or not track.is_activated:
                continue
            box = detections[di].tlwh
            observed = box[:2] + box[2:] / 2
            residuals.append(observed - track.mean[:2])
        self.counts["frames"] += 1
        self.counts["anchor_observations"] += len(residuals)
        if len(residuals) < self.min_anchors:
            self.counts["insufficient_anchor_frames"] += 1
            return
        residuals = np.asarray(residuals)
        shift = np.median(residuals, axis=0)
        dispersion = float(np.median(np.linalg.norm(residuals - shift, axis=1)) / self.diag)
        if not np.isfinite(shift).all() or not np.isfinite(dispersion) or dispersion > self.max_residual:
            self.counts["rejected_dispersion_frames"] += 1
            return
        self.shift = shift
        self.shifts.append(float(np.linalg.norm(shift)))
        self.counts["eligible_frames"] += 1

    def correct(self, lost, frame_id):
        for track in lost:
            if track.state != TrackState.Lost:
                continue
            self.counts["lost_track_frames"] += 1
            if frame_id - track.end_frame > self.ttl:
                self.counts["expired_track_frames"] += 1
                continue
            if self.shift is None:
                continue
            proposed = track.mean[:2] + self.shift
            if not (0 <= proposed[0] <= self.width and 0 <= proposed[1] <= self.height):
                self.counts["outside_image_track_frames"] += 1
                continue
            self.counts["eligible_track_frames"] += 1
            if self.mode == "apply":
                track.mean[:2] = proposed
                self.counts["corrected_track_frames"] += 1

    def save(self, path):
        path.write_text(json.dumps({
            "mode": self.mode, "counts": dict(self.counts),
            "shift_norm_median_px": float(np.median(self.shifts)) if self.shifts else None,
            "shift_norm_max_px": float(max(self.shifts)) if self.shifts else None,
        }, indent=2) + "\n")
