"""Explicit Algorithm-1-style lifecycle, separate from the ByteTrack control.

Unspecified choices: confidence of a trajectory = latest matched score;
second-pass IoU threshold=.5; discard history after T-1 unobserved frames.
High detections initialize immediately; no .7 birth or unconfirmed stage.
"""
import numpy as np
from collections import Counter
from yolox.tracker.byte_tracker import STrack, joint_stracks, remove_duplicate_stracks
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
        self.audit_counts = Counter()
        self.audit_events = []

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
        max_gap = getattr(self.args, 'mha_max_gap', homa.window-1)
        self.audit_counts['expired'] += sum(self.frame_id-t.frame_id > max_gap for t in pool)
        pool = [t for t in pool if self.frame_id - t.frame_id <= max_gap]
        STrack.multi_predict(pool)
        all_pool = getattr(self.args,'mha_all_pool',False)
        thigh = [t for t in pool if all_pool or t.score > self.args.track_thresh]
        tlow = [] if all_pool else [t for t in pool if t.score <= self.args.track_thresh]
        costs = homa.cost(thigh, dhigh)
        if getattr(self.args,'mha_score_fusion',False):
            if homa.arm != 'mha_iou':
                raise ValueError('score-fusion probe is restricted to IoU cost')
            costs = matching.fuse_score(costs,dhigh)
        matches, unmatched, ud = matching.linear_assignment(costs, thresh=self.args.match_thresh)
        self.audit_counts['stage1_matches'] += len(matches)
        # Passive counter: excluded low-score trajectory overlaps an unmatched high detection.
        # This is a lifecycle opportunity, not a GT identity-switch label.
        blocked = 1-matching.iou_distance(tlow,dhigh)
        for i,t in enumerate(tlow):
            for j in ud:
                if blocked[i,j] >= .5:
                    self.audit_counts['excluded_low_overlapping_unmatched_high'] += 1
                    if len(self.audit_events)<30:
                        self.audit_events.append({'frame':self.frame_id,'old_track_id':t.track_id,
                            'last_frame':t.frame_id,'last_score':float(t.score),'high_index':int(j),
                            'high_score':float(dhigh[j].score),'iou':float(blocked[i,j]),
                            'old_tlwh':t.tlwh.tolist(),'detection_tlwh':dhigh[j].tlwh.tolist()})
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
        self.audit_counts['stage2_matches'] += len(matches)
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
            if track.score < getattr(self.args,'mha_birth_thresh',self.args.track_thresh):
                self.audit_counts['birth_rejected'] += 1
                continue
            track.activate(self.kalman_filter, self.frame_id)
            track.is_activated = True
            active.append(track)
            self.audit_counts['births'] += 1
        if getattr(self.args,'mha_remove_duplicates',False):
            before=len(active)+len(lost)
            active,lost=remove_duplicate_stracks(active,lost)
            self.audit_counts['duplicates_removed'] += before-len(active)-len(lost)
        self.tracked_stracks, self.lost_stracks = active, lost
        homa.remember(active + lost)
        return active
