"""Explicit Algorithm-1-style lifecycle, separate from the ByteTrack control.

Unspecified choices: confidence of a trajectory = latest matched score;
second-pass IoU threshold=.5; discard history after T-1 unobserved frames.
High detections initialize immediately; no .7 birth or unconfirmed stage.
"""
import numpy as np
from yolox.tracker.byte_tracker import STrack, joint_stracks
from yolox.tracker.basetrack import TrackState
from yolox.tracker.kalman_filter import KalmanFilter
from yolox.tracker import matching


class MHATracker:
    def __init__(self, args, frame_rate=30):
        self.args = args
        self.kalman_filter = KalmanFilter()
        self.tracked_stracks = []
        self.lost_stracks = []
        self.frame_id = 0
        self.homa = None

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        homa = self.homa
        if homa is None:
            raise ValueError('MHATracker requires an association instance')
        boxes = output_results[:, :4].copy()
        scores = output_results[:, 4]
        boxes /= min(img_size[0] / float(img_info[0]), img_size[1] / float(img_info[1]))
        high = scores > self.args.track_thresh
        low = (scores > .1) & ~high
        def detections(mask):
            return [STrack(STrack.tlbr_to_tlwh(b), float(s)) for b, s in zip(boxes[mask], scores[mask])]
        dhigh, dlow = detections(high), detections(low)
        if homa.current is not None:
            if len(homa.current) != len(dhigh):
                raise ValueError('high-score feature count mismatch')
            for d, feature in zip(dhigh, homa.current):
                d.homa_feature = feature
                d.homa_feature_frame = self.frame_id
        pool = joint_stracks(self.tracked_stracks, self.lost_stracks)
        pool = [t for t in pool if self.frame_id - t.frame_id < homa.window]
        STrack.multi_predict(pool)
        thigh = [t for t in pool if t.score > self.args.track_thresh]
        tlow = [t for t in pool if t.score <= self.args.track_thresh]
        matches, unmatched, ud = matching.linear_assignment(homa.cost(thigh, dhigh), thresh=self.args.match_thresh)
        active = []
        for it, idet in matches:
            track = thigh[it]
            if track.state == TrackState.Tracked:
                track.update(dhigh[idet], self.frame_id)
            else:
                track.re_activate(dhigh[idet], self.frame_id, new_id=False)
            active.append(track)
        # Text in §3.4 restricts low-score recovery to previous-frame tracklets.
        remaining = [t for t in tlow + [thigh[i] for i in unmatched] if t.frame_id == self.frame_id - 1]
        matches, _, _ = matching.linear_assignment(matching.iou_distance(remaining, dlow), thresh=.5)
        for it, idet in matches:
            track = remaining[it]
            track.update(dlow[idet], self.frame_id)
            active.append(track)
        active_ids = {t.track_id for t in active}
        lost = []
        for t in pool:
            if t.track_id not in active_ids:
                t.mark_lost()
                lost.append(t)
        for idet in ud:
            track = dhigh[idet]
            track.activate(self.kalman_filter, self.frame_id)
            track.is_activated = True
            active.append(track)
        self.tracked_stracks, self.lost_stracks = active, lost
        homa.remember(active + lost)
        return active
