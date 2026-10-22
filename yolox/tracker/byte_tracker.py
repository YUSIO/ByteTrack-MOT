import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState


def _clone_topology(topology):
    if topology is None:
        return None
    return np.asarray(topology, dtype=np.float32).copy()


def build_local_topologies(boxes, max_neighbors=8, min_neighbors=2, version="paper"):
    """Paper Eq. 4-8; legacy preserves Exp014 for controlled replay.

    Each descriptor contains rows ``[normalized_distance, sin(theta),
    cos(theta)]`` for the nearest detections in the same frame.  The
    descriptor is intentionally identity-free: a candidate detection is
    compared with a stored track descriptor through a Hungarian assignment
    over neighbor relations.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    count = len(boxes)
    if count == 0:
        return []
    centers = np.column_stack(
        ((boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5)
    ).astype(np.float32)
    pairwise = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    nonzero = pairwise[pairwise > 0]
    frame_scale = float(nonzero.mean()) if nonzero.size else 1.0
    descriptors = []
    for index, center in enumerate(centers):
        delta = centers - center
        distances = np.linalg.norm(delta, axis=1)
        distances[index] = np.inf
        order = np.argsort(distances)
        order = order[np.isfinite(distances[order])]
        order = order[:max_neighbors]
        if len(order) < min_neighbors:
            descriptors.append(None)
            continue

        selected_distances = distances[order]
        scale = max(float(np.max(selected_distances)), 1e-6) if version == "legacy" else frame_scale
        selected_delta = delta[order]
        safe_distances = np.maximum(selected_distances, 1e-6)
        theta = np.arctan2(selected_delta[:, 1], selected_delta[:, 0])
        sin_theta = selected_delta[:, 1] / safe_distances if version == "legacy" else np.sin(theta)
        cos_theta = selected_delta[:, 0] / safe_distances if version == "legacy" else np.cos(theta)
        descriptors.append(
            np.column_stack(
                (
                    selected_distances / scale,
                    sin_theta,
                    cos_theta,
                )
            ).astype(np.float32)
        )
    return descriptors


def local_topology_cost(track_topology, detection_topology, alpha=0.6, version="paper"):
    """Compare two local topology descriptors with Hungarian matching."""
    if track_topology is None or detection_topology is None:
        return None
    track_topology = np.asarray(track_topology, dtype=np.float32)
    detection_topology = np.asarray(detection_topology, dtype=np.float32)
    if track_topology.ndim != 2 or detection_topology.ndim != 2:
        return None
    if len(track_topology) == 0 or len(detection_topology) == 0:
        return None

    distance_cost = np.abs(
        track_topology[:, None, 0] - detection_topology[None, :, 0]
    )
    direction_dot = np.sum(
        track_topology[:, None, 1:3] * detection_topology[None, :, 1:3], axis=2
    )
    direction_cost = 0.5 * (1.0 - np.clip(direction_dot, -1.0, 1.0))
    if version != "legacy":
        direction_cost = np.linalg.norm(track_topology[:, None, 1:3] - detection_topology[None, :, 1:3], axis=2) / 2.0
    pair_cost = alpha * distance_cost + (1.0 - alpha) * direction_cost

    rows, columns = linear_sum_assignment(pair_cost)
    if version != "legacy":
        return float(np.clip(pair_cost[rows, columns].mean() + 0.1 * abs(len(track_topology) - len(detection_topology)), 0.0, 1.0))
    unmatched = max(len(track_topology), len(detection_topology)) - len(rows)
    return float((pair_cost[rows, columns].sum() + unmatched) / max(len(track_topology), len(detection_topology)))


def topology_fused_cost(iou_cost, tracks, detections, alpha=0.6, topology_lambda=0.30, version="paper"):
    """Fuse topology with IoU only where both local descriptors are valid."""
    fused = np.asarray(iou_cost, dtype=np.float32).copy()
    if fused.size == 0:
        return fused
    topology_lambda = float(np.clip(topology_lambda, 0.0, 1.0))
    for track_index, track in enumerate(tracks):
        for detection_index, detection in enumerate(detections):
            topology_cost = local_topology_cost(
                getattr(track, "topology", None),
                getattr(detection, "topology", None),
                alpha=alpha,
                version=version,
            )
            if topology_cost is None:
                continue
            fused[track_index, detection_index] = (
                (1.0 - topology_lambda) * fused[track_index, detection_index]
                + topology_lambda * topology_cost
            )
    return fused

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, topology=None, topology_version="paper"):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.topology_version = topology_version
        self.topology = _clone_topology(topology)
        self.topology_frame_id = None

    def update_topology(self, topology, frame_id):
        if topology is not None or getattr(self, "topology_version", "paper") != "legacy":
            self.topology = _clone_topology(topology)
            self.topology_frame_id = frame_id

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
        if self.topology is not None:
            self.topology_frame_id = frame_id

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
        self.update_topology(new_track.topology, frame_id)

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
        self.update_topology(new_track.topology, frame_id)

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
        self.topology_version = getattr(args, "topology_version", "paper")
        self.first_metric = getattr(args, "first_metric", "iou")
        self.geometry = getattr(args, "geometry", "hybrid")
        self.first_score_fusion = getattr(args, "first_score_fusion", "on") == "on" and not args.mot20
        self.audit_first = bool(getattr(args, "audit_first", False))
        self.first_audit = {k: 0 for k in ("frames", "pairs", "raw_order_violations", "fused_order_violations", "iou_eligible", "diou_eligible", "iou_matches", "iou_matches_rejected_by_diou", "shadow_changed_frames", "raw_iou_sum", "raw_diou_sum", "fused_iou_sum", "fused_diou_sum")}
        self.first_audit_examples = []
        self.topology_enabled = bool(getattr(args, "topology", False))
        self.topology_kmax = int(getattr(args, "topology_kmax", 8))
        self.topology_kmin = int(getattr(args, "topology_kmin", 2))
        self.topology_alpha = float(getattr(args, "topology_alpha", 0.6))
        self.topology_lambda = float(getattr(args, "topology_lambda", 0.30))

    def audit_first_costs(self, tracks, detections):
        """Read-only shadow assignments on the SAME predicted state; never use GT."""
        raw_iou = matching.iou_distance(tracks, detections, geometry=self.geometry)
        raw_diou = matching.diou_distance(tracks, detections, geometry=self.geometry)
        a = matching.fuse_score(raw_iou, detections) if self.first_score_fusion else raw_iou
        b = matching.fuse_score(raw_diou, detections) if self.first_score_fusion else raw_diou
        threshold = self.args.match_thresh
        ma, _, _ = matching.linear_assignment(a, threshold)
        mb, _, _ = matching.linear_assignment(b, threshold)
        sa = set(map(tuple, ma)); sb = set(map(tuple, mb))
        rejected = [(int(i), int(j)) for i, j in ma if b[i, j] > threshold]
        values = dict(frames=1, pairs=a.size,
            raw_order_violations=int(np.sum(raw_diou < raw_iou - 1e-12)),
            fused_order_violations=int(np.sum(b < a - 1e-12)),
            iou_eligible=int(np.sum(a <= threshold)), diou_eligible=int(np.sum(b <= threshold)),
            iou_matches=len(ma), iou_matches_rejected_by_diou=len(rejected),
            shadow_changed_frames=int(sa != sb), raw_iou_sum=float(raw_iou.sum()),
            raw_diou_sum=float(raw_diou.sum()), fused_iou_sum=float(a.sum()), fused_diou_sum=float(b.sum()))
        for key, value in values.items():
            self.first_audit[key] += value
        if rejected and len(self.first_audit_examples) < 3:
            self.first_audit_examples.append(dict(frame=self.frame_id, rejected=[dict(
                track_id=int(tracks[i].track_id), detection_index=j,
                prediction=tracks[i].tlbr.tolist(), detection=detections[j].tlbr.tolist(),
                score=float(detections[j].score), iou_cost=float(a[i,j]), diou_cost=float(b[i,j])) for i,j in rejected]))

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
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
        if self.topology_enabled:
            topology_indices = np.flatnonzero(inds_low)
            topology_descriptors = build_local_topologies(
                bboxes[inds_low],
                max_neighbors=self.topology_kmax,
                min_neighbors=self.topology_kmin,
                version=self.topology_version,
            )
            topology_by_detection_index = {
                int(index): descriptor
                for index, descriptor in zip(topology_indices, topology_descriptors)
            }
        else:
            topology_by_detection_index = {}
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detection_indices = np.flatnonzero(remain_inds)
            detections = [
                STrack(
                    STrack.tlbr_to_tlwh(tlbr),
                    score,
                    topology=topology_by_detection_index.get(int(index)),
                    topology_version=self.topology_version,
                )
                for index, tlbr, score in zip(detection_indices, dets, scores_keep)
            ]
        else:
            detections = []

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
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        if self.first_metric == "iou":
            dists = matching.iou_distance(strack_pool, detections, geometry=self.geometry)
        else:
            dists = matching.diou_distance(strack_pool, detections, legacy_continuous=self.first_metric == "legacy_diou", geometry=self.geometry)
        if self.first_score_fusion:
            dists = matching.fuse_score(dists, detections)
        if self.first_metric == "diou_gate":
            # Diagnostic intervention: DIoU eligibility, IoU assignment objective.
            iou_cost = matching.iou_distance(strack_pool, detections, geometry=self.geometry)
            if self.first_score_fusion:
                iou_cost = matching.fuse_score(iou_cost, detections)
            dists = np.where(dists <= self.args.match_thresh, iou_cost, np.inf)
        if self.audit_first:
            self.audit_first_costs(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

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
            detection_indices_second = np.flatnonzero(inds_second)
            detections_second = [
                STrack(
                    STrack.tlbr_to_tlwh(tlbr),
                    score,
                    topology=topology_by_detection_index.get(int(index)),
                    topology_version=self.topology_version,
                )
                for index, tlbr, score in zip(
                    detection_indices_second, dets_second, scores_second
                )
            ]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second, geometry=self.geometry)
        if self.topology_enabled:
            dists = topology_fused_cost(
                dists,
                r_tracked_stracks,
                detections_second,
                alpha=self.topology_alpha,
                topology_lambda=self.topology_lambda,
                version=self.topology_version,
            )
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
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
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections, geometry=self.geometry)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
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
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
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
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks, geometry=self.geometry)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

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


def remove_duplicate_stracks(stracksa, stracksb, geometry="hybrid"):
    pdist = matching.iou_distance(stracksa, stracksb, geometry=geometry)
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
