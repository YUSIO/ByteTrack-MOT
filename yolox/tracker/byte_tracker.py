import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def record_prediction(self, max_len):
        """Keep the latest one-step Kalman predictions (centre and positional std)."""
        if not hasattr(self, 'prediction_history') or self.prediction_history.maxlen != max_len:
            self.prediction_history = deque(getattr(self, 'prediction_history', []), maxlen=max_len)
        std = np.sqrt(np.diag(self.covariance)[:2])
        self.prediction_history.append(np.array([self.mean[0], self.mean[1], std[0], std[1]]))

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, warp):
        """Move predicted states by the camera motion (BoT-SORT GMC, written for the xyah state; as in Exp055 9b9b161).

        ``warp`` is the 2x3 similarity mapping image points of the previous frame to the current one.
        Centre and velocity go through its linear part, height and its velocity through its scale;
        the covariance is transformed with the same Jacobian.
        """
        linear, shift = warp[:, :2], warp[:, 2]
        scale = float(np.sqrt(abs(np.linalg.det(linear))))
        jac = np.eye(8)
        jac[0:2, 0:2] = linear
        jac[4:6, 4:6] = linear
        jac[3, 3] = jac[7, 7] = scale
        for st in stracks:
            st.mean = jac @ st.mean
            st.mean[:2] += shift
            st.covariance = jac @ st.covariance @ jac.T

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        if hasattr(new_track, "homa_feature"):
            self.homa_feature = new_track.homa_feature
            self.homa_feature_frame = frame_id

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if hasattr(new_track, "homa_feature"):
            self.homa_feature = new_track.homa_feature
            self.homa_feature_frame = frame_id

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        trace = getattr(self, "decision_trace", None)
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        if trace is not None:
            high_indices, low_indices = np.flatnonzero(remain_inds), np.flatnonzero(inds_second)
            trace.record("detection_bands", boxes_xyxy=bboxes, scores=scores,
                         high_indices=high_indices, low_indices=low_indices,
                         ignored_indices=np.flatnonzero(~(remain_inds | inds_second)),
                         track_thresh=self.args.track_thresh, low_thresh=0.1,
                         birth_thresh=self.det_thresh, max_time_lost=self.max_time_lost)

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        homa = getattr(self, 'homa', None)
        if homa is not None and homa.current is not None:
            if len(homa.current) != len(detections):
                raise ValueError('high-score feature count mismatch')
            for det, feature in zip(detections, homa.current):
                det.homa_feature = feature
                det.homa_feature_frame = self.frame_id

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        if getattr(self.args, 'freeze_lost', False):
            # Exp074: a Lost track stays where it was last seen (in the image, or in the scene with GMC) instead of
            # drifting on its last velocity; the velocity is re-estimated from the next matched box.
            for track in self.lost_stracks:
                track.mean[4:6] = 0
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        warp = getattr(self.args, 'gmc_warps', {}).get(self.frame_id)
        if warp is not None:
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)
        # Default 'iou' is upstream ByteTrack; 'm2da' follows HOMATracker Eqs. (9)-(13).
        first_stage_cost = getattr(self.args, 'first_stage_cost', 'iou')
        if homa is not None:
            dists = homa.cost(strack_pool, detections)
        elif first_stage_cost == 'iou':
            dists = matching.iou_distance(strack_pool, detections)
        elif first_stage_cost == 'm2da':
            for track in strack_pool:
                track.record_prediction(self.args.m2da_window - 1)
            dists = matching.m2da_distance(strack_pool, detections, self.args.m2da_kappa)
        else:
            raise ValueError('unknown first_stage_cost: {}'.format(first_stage_cost))
        if homa is None and not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        if trace is not None:
            trace.association("association_high", strack_pool, detections, high_indices, dists,
                              matches, u_track, u_detection, self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        # Exp070 diagnostic (off by default): tracks lost for at most `low_score_lost` frames also take low score boxes
        k = getattr(self.args, "low_score_lost", 0)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked
                             or (k > 0 and strack_pool[i].state == TrackState.Lost and self.frame_id - strack_pool[i].end_frame <= k)]
        if trace is not None:
            eligible_objects = set(r_tracked_stracks)
            trace.record("low_eligibility", low_score_lost=k,
                         tracks=[dict(track=trace.track(strack_pool[i]), eligible=strack_pool[i] in eligible_objects,
                                      reason="eligible" if strack_pool[i] in eligible_objects else "state_or_lost_age",
                                      age=self.frame_id - strack_pool[i].end_frame) for i in u_track])
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        if trace is not None:
            trace.association("association_low", r_tracked_stracks, detections_second, low_indices, dists,
                              matches, u_track, u_detection_second, 0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        if trace is not None:
            remaining_high_indices = high_indices[list(u_detection)]
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        if trace is not None:
            trace.association("association_unconfirmed", unconfirmed, detections, remaining_high_indices, dists,
                              matches, u_unconfirmed, u_detection, 0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                if trace is not None:
                    trace.record("byte_birth", detection_index=int(remaining_high_indices[inew]),
                                 score=track.score, decision="below_det_thresh")
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
            if trace is not None:
                trace.record("byte_birth", detection_index=int(remaining_high_indices[inew]),
                             track=trace.track(track), decision="activated")
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                if trace is not None:
                    trace.record("byte_remove", track=trace.track(track), reason="max_time_lost",
                                 age=self.frame_id - track.end_frame, max_time_lost=self.max_time_lost)
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        if trace is not None:
            duplicate_before = trace.pools(self)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        if trace is not None:
            trace.record("byte_deduplicate", before=duplicate_before, after=trace.pools(self))
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        if homa is not None:
            homa.remember(self.tracked_stracks + self.lost_stracks)
        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
