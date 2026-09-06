#!/usr/bin/env python3
"""Train and apply GT-supervised affinity between a ByteTrack state and a detection.

Labels are the current GT identity relation of an existing tracker state and a
detector box.  Baseline tracker states are inputs only; no tracker-success label
or future-frame feature is used.  At inference, the model adjusts only primary
association costs for already-existing tracks and never changes detector scores
or the new-track birth rule.
"""

import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from causal_lineage import (
    MLP,
    binary_auc,
    iou_matrix,
    logit,
    match_ids,
    predict_logits,
    read_detections,
    read_gt,
    read_seqinfo,
    scale,
    scale_fit,
    sha256,
    tlwh_to_tlbr,
)
from yolox.tracker.basetrack import BaseTrack, TrackState
from yolox.tracker.byte_tracker import BYTETracker


AFFINITY_NAMES = (
    "track_score", "detection_score", "detection_logit", "predicted_iou",
    "center_dx", "center_dy", "log_width_ratio", "log_height_ratio",
    "frames_since_update", "tracklet_length", "track_is_active", "detection_log_area",
)


def safe_sigmoid(values):
    values = np.asarray(values, dtype=np.float32)
    positive = values >= 0.0
    result = np.empty_like(values, dtype=np.float32)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _tlwh_array(items):
    if not items:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray([item.tlwh for item in items], dtype=np.float32)


def _pair_iou(tracks, detections):
    return iou_matrix(tlwh_to_tlbr(_tlwh_array(tracks)), tlwh_to_tlbr(_tlwh_array(detections)))


def select_pairs(tracks, detections, track_gt_ids, detection_gt_ids, top_k):
    """Return a compact, recall-preserving set of state--detection candidate pairs."""
    if not tracks or not detections:
        return []
    overlap = _pair_iou(tracks, detections)
    track_boxes, detection_boxes = _tlwh_array(tracks), _tlwh_array(detections)
    track_centers = track_boxes[:, :2] + track_boxes[:, 2:] * 0.5
    detection_centers = detection_boxes[:, :2] + detection_boxes[:, 2:] * 0.5
    distance = np.linalg.norm(track_centers[:, None] - detection_centers[None, :], axis=-1)
    pairs, keep = set(), min(top_k, len(detections))
    for track_index, track in enumerate(tracks):
        by_iou = np.argsort(-overlap[track_index], kind="mergesort")[:keep]
        by_distance = np.argsort(distance[track_index], kind="mergesort")[:keep]
        for detection_index in set(by_iou.tolist() + by_distance.tolist()):
            pairs.add((track_index, detection_index))
        gt_identity = track_gt_ids.get(track.track_id, -1)
        if gt_identity > 0:
            for detection_index, detection_identity in enumerate(detection_gt_ids):
                if int(detection_identity) == gt_identity:
                    pairs.add((track_index, detection_index))
    return sorted(pairs)


def affinity_features(tracks, detections, pairs, width, height, frame_id):
    if not pairs:
        return np.zeros((0, len(AFFINITY_NAMES)), dtype=np.float32)
    track_boxes, detection_boxes = _tlwh_array(tracks), _tlwh_array(detections)
    overlap = _pair_iou(tracks, detections)
    rows = []
    for track_index, detection_index in pairs:
        track, detection = tracks[track_index], detections[detection_index]
        track_box, detection_box = track_boxes[track_index], detection_boxes[detection_index]
        track_center = track_box[:2] + track_box[2:] * 0.5
        detection_center = detection_box[:2] + detection_box[2:] * 0.5
        track_score, detection_score = float(track.score), float(detection.score)
        rows.append((
            track_score,
            detection_score,
            logit([detection_score])[0],
            overlap[track_index, detection_index],
            (detection_center[0] - track_center[0]) / max(float(width), 1.0),
            (detection_center[1] - track_center[1]) / max(float(height), 1.0),
            math.log(max(float(detection_box[2]), 1e-3) / max(float(track_box[2]), 1e-3)),
            math.log(max(float(detection_box[3]), 1e-3) / max(float(track_box[3]), 1e-3)),
            min(max(int(frame_id) - int(track.frame_id), 0), 30) / 30.0,
            min(max(int(track.tracklet_len), 0), 30) / 30.0,
            float(track.state == TrackState.Tracked),
            math.log(max(float(detection_box[2] * detection_box[3]) / float(width * height), 1e-8)),
        ))
    return np.asarray(rows, dtype=np.float32)


def _tracker_args(track_thresh, det_thresh, track_buffer, match_thresh):
    return SimpleNamespace(
        track_thresh=track_thresh,
        det_thresh=det_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        mot20=False,
        association_observer=None,
        association_adjuster=None,
    )


def _match_output_tracks(tracks, gt_rows):
    if not tracks:
        return {}
    rows = np.asarray([list(track.tlwh) + [float(track.score)] for track in tracks], dtype=np.float32)
    _, identities = match_ids(rows, gt_rows)
    return {
        track.track_id: int(identity)
        for track, identity in zip(tracks, identities)
        if int(identity) > 0
    }


def extract_track_affinity_supervision(
    dataset_root,
    detections_root,
    sequences,
    split="train",
    top_k=5,
    track_thresh=0.6,
    det_thresh=0.7,
    track_buffer=30,
    match_thresh=0.9,
):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    all_features, all_labels = [], []
    summary = {
        "split": split,
        "pair_top_k": top_k,
        "label": "existing_tracker_state_and_current_detection_share_gt_identity",
        "gt_iou_threshold": 0.5,
        "tracker_state_source": "frozen_detector_cache_with_unmodified_bytetrack_before_primary_association",
        "tracker_parameters": {
            "track_thresh": track_thresh,
            "det_thresh": det_thresh,
            "track_buffer": track_buffer,
            "match_thresh": match_thresh,
        },
        "sequences": {},
    }
    for sequence in sequences:
        sequence_dir = dataset_root / split / sequence
        detection_path = detections_root / sequence / "det.txt"
        if not sequence_dir.is_dir() or not detection_path.is_file():
            raise FileNotFoundError("missing sequence or detector cache for {}".format(sequence))
        info = read_seqinfo(sequence_dir / "seqinfo.ini")
        detections = read_detections(detection_path)
        ground_truth = read_gt(sequence_dir / "gt" / "gt.txt")
        args = _tracker_args(track_thresh, det_thresh, track_buffer, match_thresh)
        tracker = BYTETracker(args, frame_rate=30)
        track_gt_ids, sequence_features, sequence_labels = {}, [], []

        for frame_id in range(1, info["frames"] + 1):
            current = detections.get(frame_id, np.zeros((0, 5), dtype=np.float32))
            high = current[current[:, 4] > track_thresh]
            _, current_gt_ids = match_ids(high, ground_truth.get(frame_id, np.zeros((0, 5), dtype=np.float32)))

            def observe(stage, tracks, current_detections, current_frame):
                if stage != "primary" or not tracks or not current_detections:
                    return
                pairs = select_pairs(tracks, current_detections, track_gt_ids, current_gt_ids, top_k)
                if not pairs:
                    return
                sequence_features.append(
                    affinity_features(tracks, current_detections, pairs, info["width"], info["height"], current_frame)
                )
                sequence_labels.append(np.asarray([
                    float(track_gt_ids.get(tracks[a].track_id, -1) > 0 and track_gt_ids.get(tracks[a].track_id, -1) == int(current_gt_ids[b]))
                    for a, b in pairs
                ], dtype=np.float32))

            args.association_observer = observe
            outputs = np.column_stack((current[:, :2], current[:, :2] + current[:, 2:4], current[:, 4])) if len(current) else np.zeros((0, 5), dtype=np.float32)
            online_tracks = tracker.update(outputs, (info["height"], info["width"]), (info["height"], info["width"]))
            track_gt_ids.update(_match_output_tracks(online_tracks, ground_truth.get(frame_id, np.zeros((0, 5), dtype=np.float32))))

        features = np.concatenate(sequence_features).astype(np.float32) if sequence_features else np.zeros((0, len(AFFINITY_NAMES)), dtype=np.float32)
        labels = np.concatenate(sequence_labels).astype(np.float32) if sequence_labels else np.zeros((0,), dtype=np.float32)
        all_features.append(features)
        all_labels.append(labels)
        summary["sequences"][sequence] = {
            "frames": info["frames"],
            "pair_samples": int(len(labels)),
            "pair_positive": int(labels.sum()),
            "det_sha256": sha256(detection_path),
            "gt_sha256": sha256(sequence_dir / "gt" / "gt.txt"),
        }
    data = {
        "x": np.concatenate(all_features).astype(np.float32),
        "y": np.concatenate(all_labels).astype(np.float32),
    }
    summary["totals"] = {"pair_samples": int(len(data["y"])), "pair_positive": int(data["y"].sum())}
    return data, summary


def affinity_metrics(model, values, labels, device):
    probability = safe_sigmoid(predict_logits(model, values, device))
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return {
        "bce": float(-(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)).mean()),
        "auc": binary_auc(probability, labels),
        "positive_rate": float(labels.mean()),
    }


def fit_track_affinity(fit, validation, output_dir, device_name, seed=0, epochs=40, hidden_dim=64, dropout=0.1, learning_rate=1e-3, batch_size=4096):
    if min(len(fit["y"]), len(validation["y"])) == 0 or not fit["y"].sum() or not validation["y"].sum():
        raise ValueError("fit and validation affinity supervision must contain positive and negative samples")
    torch.manual_seed(seed)
    generator, device = np.random.default_rng(seed), torch.device(device_name)
    mean, std = scale_fit(fit["x"])
    fit_x, validation_x = scale(fit["x"], mean, std), scale(validation["x"], mean, std)
    model = MLP(len(AFFINITY_NAMES), hidden_dim, dropout).to(device)
    positive_weight = min((len(fit["y"]) - fit["y"].sum()) / max(fit["y"].sum(), 1.0), 30.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    checkpoint, history, best = output_dir / "checkpoint.pt", [], {"bce": float("inf"), "epoch": -1}
    for epoch in range(1, epochs + 1):
        model.train()
        order = generator.permutation(len(fit_x))
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(torch.from_numpy(fit_x[indices]).to(device)), torch.from_numpy(fit["y"][indices]).to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        metric = affinity_metrics(model, validation_x, validation["y"], device)
        history.append({"epoch": epoch, "validation": metric})
        if metric["bce"] < best["bce"]:
            best = {"bce": metric["bce"], "epoch": epoch}
            torch.save({
                "schema_version": 1,
                "method": "track_conditioned_affinity",
                "feature_names": AFFINITY_NAMES,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "state": model.state_dict(),
                "mean": mean,
                "std": std,
                "selected_epoch": epoch,
                "seed": seed,
            }, checkpoint)
    return {"checkpoint": str(checkpoint), "best": best, "epochs": history}


class TrackAffinityPredictor:
    def __init__(self, checkpoint_path, device_name, width, height, weight, minimum_probability):
        if not (0.0 <= weight <= 1.0 and 0.0 < minimum_probability < 1.0):
            raise ValueError("invalid affinity inference parameters")
        self.device = torch.device(device_name)
        payload = torch.load(checkpoint_path, map_location=self.device)
        if payload.get("schema_version") != 1 or payload.get("method") != "track_conditioned_affinity":
            raise ValueError("unsupported affinity checkpoint: {}".format(checkpoint_path))
        self.model = MLP(len(AFFINITY_NAMES), payload["hidden_dim"], payload["dropout"]).to(self.device)
        self.model.load_state_dict(payload["state"])
        self.model.eval()
        self.mean, self.std = np.asarray(payload["mean"], dtype=np.float32), np.asarray(payload["std"], dtype=np.float32)
        self.width, self.height, self.weight, self.minimum_probability = width, height, weight, minimum_probability

    def __call__(self, stage, tracks, detections, costs, frame_id):
        if stage != "primary" or not tracks or not detections or self.weight == 0.0:
            return costs
        pairs = [(track_index, detection_index) for track_index in range(len(tracks)) for detection_index in range(len(detections))]
        values = affinity_features(tracks, detections, pairs, self.width, self.height, frame_id)
        probability = safe_sigmoid(predict_logits(self.model, scale(values, self.mean, self.std), self.device)).reshape(len(tracks), len(detections))
        blended = (1.0 - self.weight) * costs + self.weight * (1.0 - probability)
        return np.where(probability >= self.minimum_probability, blended, costs).astype(np.float32)
