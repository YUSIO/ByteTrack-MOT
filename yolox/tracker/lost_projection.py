"""Neighbour-projected state maintenance for Lost tracks (Exp044).

While a track is Lost, ByteTrack coasts it with the Kalman constant-velocity
model. This module replaces or corrects that coasting position with a
projection from anchors: tracks that were Tracked and matched to a high-score
detection in the first association of the current frame.

The correction is computed after association at frame t and therefore only
affects the position used for association at frame t + 1. No box is output,
no detection is added and the track lifecycle is untouched.

``mode='shadow'`` runs the same recursion without writing to the Kalman state,
so the tracker output stays identical to the unmodified tracker while the
trace records where the projected state would have been.
"""
import json

import numpy as np

from .basetrack import TrackState


def similarity_project(point, src, dst, diag, max_residual):
    """Fit dst ~ s * R * src + t on anchors and map ``point``. Returns (xy, residual) or None."""
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    denom = float((src_c ** 2).sum())
    if denom <= 1e-12:
        return None
    u, sing, vt = np.linalg.svd(src_c.T @ dst_c)
    rot = vt.T @ u.T
    sign = 1.0
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = vt.T @ u.T
        sign = -1.0
    scale = float((sing[0] + sign * sing[1]) / denom)
    residual = float(np.median(np.linalg.norm(dst_c - scale * (src_c @ rot.T), axis=1)) / diag)
    if residual > max_residual:
        return None
    return scale * ((point - src_mean) @ rot.T) + dst_mean, residual


def translation_project(point, src, dst, diag, max_residual):
    """Shift ``point`` by the component-wise median anchor displacement. Returns (xy, residual) or None."""
    disp = dst - src
    shift = np.median(disp, axis=0)
    residual = float(np.median(np.linalg.norm(disp - shift, axis=1)) / diag)
    if residual > max_residual:
        return None
    return point + shift, residual


class LostProjector(object):
    def __init__(self, mode, base, transform, neighbors, ttl, min_anchors, max_residual,
                 img_w, img_h, trace_path=None):
        if mode not in ("shadow", "apply"):
            raise ValueError("mode must be shadow or apply")
        if base not in ("innov", "follow"):
            raise ValueError("base must be innov or follow")
        if transform not in ("sim", "trans"):
            raise ValueError("transform must be sim or trans")
        if base == "follow" and transform != "trans":
            raise ValueError("base=follow supports transform=trans only")
        self.mode, self.base, self.transform = mode, base, transform
        self.neighbors, self.ttl = int(neighbors), int(ttl)
        self.min_anchors, self.max_residual = int(min_anchors), float(max_residual)
        self.img_w, self.img_h = float(img_w), float(img_h)
        self.diag = float(np.hypot(img_w, img_h))
        self._trace = open(trace_path, "w") if trace_path is not None else None
        self._state = {}  # track_id -> [center, velocity, applied_frames]
        self._prev, self._pred, self._proj_pred = {}, {}, {}
        self._last_det = {}  # track_id -> (frame_id, centre of its first-association detection)
        self._anchors = (np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 2)))
        self._frame_last_det, self._prev_last_det = {}, {}

    def close(self):
        if self._trace is not None:
            self._trace.close()
            self._trace = None

    def _write(self, row):
        if self._trace is not None:
            self._trace.write(json.dumps(row) + "\n")

    def begin_frame(self, strack_pool):
        """Record posterior centres of frame t - 1, before the Kalman prediction."""
        self._prev = {t.track_id: t.mean[:2].copy() for t in strack_pool}

    def after_predict(self, strack_pool, frame_id):
        """Record the positions used for association at frame t."""
        self._pred, self._proj_pred = {}, {}
        for track in strack_pool:
            kf_xy = track.mean[:2].copy()
            self._pred[track.track_id] = kf_xy
            state = self._state.get(track.track_id) if track.state == TrackState.Lost else None
            proj_xy = state[0] + state[1] if state is not None else kf_xy
            self._proj_pred[track.track_id] = proj_xy
            if track.state == TrackState.Lost:
                x, y, w, h = track.tlwh
                self._write({
                    "type": "pred", "frame": frame_id, "track_id": track.track_id,
                    "gap": frame_id - track.end_frame,
                    "kf_tlwh": [float(x), float(y), float(w), float(h)],
                    "proj_tlwh": [float(proj_xy[0] - w / 2), float(proj_xy[1] - h / 2), float(w), float(h)],
                    "applied_frames": state[2] if state is not None else 0,
                })

    def after_first_association(self, strack_pool, detections, matches, frame_id):
        """Collect anchors before the matched tracks are updated.

        ``innov`` anchors pair the Kalman prediction with the matched detection.
        ``follow`` anchors pair the detection matched at frame t - 1 with the one
        matched at frame t, so they need a first-association match in both frames.
        """
        prev, pred, det = [], [], []
        last_det = {}
        for itrack, idet in matches:
            track = strack_pool[itrack]
            x, y, w, h = detections[idet].tlwh
            center = np.array([x + w / 2, y + h / 2], dtype=float)
            last_det[track.track_id] = (frame_id, center)
            if track.state != TrackState.Tracked:
                continue
            before = self._last_det.get(track.track_id)
            if self.base == "follow":
                if before is None or before[0] != frame_id - 1:
                    continue
                prev.append(before[1])
            else:
                prev.append(self._prev[track.track_id])
            pred.append(self._pred[track.track_id])
            det.append(center)
        self._frame_last_det, self._prev_last_det = last_det, self._last_det
        shape = (len(det), 2)
        self._anchors = (np.asarray(prev, dtype=float).reshape(shape),
                         np.asarray(pred, dtype=float).reshape(shape),
                         np.asarray(det, dtype=float).reshape(shape))

    def _correct(self, pred_xy, prev_xy):
        prev, pred, det = self._anchors
        src = pred if self.base == "innov" else prev
        point = pred_xy if self.base == "innov" else prev_xy
        if len(src) < self.min_anchors:
            return None
        if self.neighbors > 0 and len(src) > self.neighbors:
            keep = np.argsort(np.linalg.norm(src - point, axis=1), kind="stable")[:self.neighbors]
            src, det = src[keep], det[keep]
        fit = similarity_project if self.transform == "sim" else translation_project
        result = fit(point, src, det, self.diag, self.max_residual)
        if result is None:
            return None
        xy, residual = result
        if not (0.0 <= xy[0] <= self.img_w and 0.0 <= xy[1] <= self.img_h):
            return None
        return xy, residual, len(src)

    def end_frame(self, lost_stracks, frame_id):
        """Correct the state of every track that stays Lost after frame t."""
        new_state = {}
        for track in lost_stracks:
            tid = track.track_id
            if tid not in self._proj_pred:
                continue
            state = self._state.get(tid)
            pred_xy = self._proj_pred[tid]
            if state is not None:
                prev_xy = state[0]
            else:
                before = self._prev_last_det.get(tid)
                prev_xy = before[1] if before is not None and before[0] == frame_id - 1 else self._prev[tid]
            velocity = state[1] if state is not None else track.mean[4:6].copy()
            applied = state[2] if state is not None else 0
            gap = frame_id - track.end_frame
            center, residual, n_anchors = pred_xy, None, 0
            if gap <= self.ttl:
                result = self._correct(pred_xy, prev_xy)
                if result is not None:
                    center, residual, n_anchors = result
                    applied += 1
                    if self.base == "follow":
                        velocity = center - prev_xy
            new_state[tid] = [np.asarray(center, dtype=float), np.asarray(velocity, dtype=float), applied]
            if self.mode == "apply":
                track.mean[0:2] = center
                if self.base == "follow":
                    track.mean[4:6] = velocity
            self._write({"type": "corr", "frame": frame_id, "track_id": tid, "gap": gap,
                         "applied": residual is not None, "anchors": n_anchors, "residual": residual})
        self._state = new_state
        self._last_det = self._frame_last_det
