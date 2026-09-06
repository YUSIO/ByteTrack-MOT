#!/usr/bin/env python3
"""GT-supervised, GT-free-at-inference causal low-score lineage rescorer."""

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


NODE_NAMES = (
    "score", "logit", "log_area", "log_aspect", "center_x", "center_y",
    "score_rank", "frame_count", "nearest_distance", "previous_max_iou", "previous_best_score",
)
EDGE_NAMES = (
    "previous_score", "current_score", "previous_logit", "current_logit",
    "center_dx", "center_dy", "log_width_ratio", "log_height_ratio", "iou", "score_delta",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logit(values):
    values = np.clip(np.asarray(values, dtype=np.float32), 1e-5, 1.0 - 1e-5)
    return np.log(values / (1.0 - values))


def tlwh_to_tlbr(boxes):
    result = np.asarray(boxes, dtype=np.float32).copy()
    if len(result):
        result[:, 2] += result[:, 0]
        result[:, 3] += result[:, 1]
    return result


def iou_matrix(first, second):
    if len(first) == 0 or len(second) == 0:
        return np.zeros((len(first), len(second)), dtype=np.float32)
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection_size = np.maximum(bottom_right - top_left, 0.0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    first_area = (first[:, 2] - first[:, 0]) * (first[:, 3] - first[:, 1])
    second_area = (second[:, 2] - second[:, 0]) * (second[:, 3] - second[:, 1])
    union = first_area[:, None] + second_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0).astype(np.float32)


def read_seqinfo(path):
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = raw.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    required = ("seqLength", "imWidth", "imHeight")
    if any(key not in values for key in required):
        raise ValueError("{}: missing required sequence metadata".format(path))
    return {"frames": int(values["seqLength"]), "width": int(values["imWidth"]), "height": int(values["imHeight"])}


def read_detections(path):
    frames = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            fields = raw.rstrip("\n").split(",")
            if len(fields) < 7:
                raise ValueError("{}:{}: expected MOT detection row".format(path, line_number))
            frame = int(float(fields[0]))
            x, y, width, height, score = [float(value) for value in fields[2:7]]
            if frame < 1 or width <= 0 or height <= 0 or not all(math.isfinite(value) for value in (x, y, width, height, score)):
                raise ValueError("{}:{}: invalid detection".format(path, line_number))
            frames[frame].append((x, y, width, height, score))
    return {frame: np.asarray(rows, dtype=np.float32) for frame, rows in frames.items()}


def read_gt(path):
    frames = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            fields = raw.rstrip("\n").split(",")
            if len(fields) < 6:
                raise ValueError("{}:{}: expected MOT GT row".format(path, line_number))
            confidence = float(fields[6]) if len(fields) > 6 else 1.0
            if confidence < 1.0:
                continue
            frame, identity = int(float(fields[0])), int(float(fields[1]))
            x, y, width, height = [float(value) for value in fields[2:6]]
            if frame < 1 or identity < 1 or width <= 0 or height <= 0:
                raise ValueError("{}:{}: invalid GT row".format(path, line_number))
            frames[frame].append((identity, x, y, width, height))
    return {frame: np.asarray(rows, dtype=np.float32) for frame, rows in frames.items()}


def node_features(current, previous, width, height):
    count = len(current)
    if count == 0:
        return np.zeros((0, len(NODE_NAMES)), dtype=np.float32)
    boxes = tlwh_to_tlbr(current[:, :4])
    score = current[:, 4]
    box_width, box_height = np.maximum(current[:, 2], 1e-3), np.maximum(current[:, 3], 1e-3)
    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    diagonal = max(math.hypot(width, height), 1.0)
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    distances[np.diag_indices_from(distances)] = np.inf
    nearest = np.min(distances, axis=1) if count > 1 else np.full(count, diagonal, dtype=np.float32)
    ranks = np.empty(count, dtype=np.float32)
    ranks[np.argsort(-score, kind="mergesort")] = np.arange(count, dtype=np.float32)
    if len(previous):
        overlaps = iou_matrix(boxes, tlwh_to_tlbr(previous[:, :4]))
        best = np.argmax(overlaps, axis=1)
        previous_iou = overlaps[np.arange(count), best]
        previous_score = previous[best, 4]
    else:
        previous_iou = np.zeros(count, dtype=np.float32)
        previous_score = np.zeros(count, dtype=np.float32)
    return np.column_stack((
        score, logit(score), np.log(np.maximum(box_width * box_height / float(width * height), 1e-8)),
        np.log(box_width / box_height), centers[:, 0] / width, centers[:, 1] / height,
        ranks / max(count - 1, 1), np.full(count, min(count, 100) / 100.0), nearest / diagonal,
        previous_iou, previous_score,
    )).astype(np.float32)


def candidate_pairs(previous, current, top_k=3):
    if len(previous) == 0 or len(current) == 0:
        return []
    previous_boxes, current_boxes = tlwh_to_tlbr(previous[:, :4]), tlwh_to_tlbr(current[:, :4])
    overlap = iou_matrix(current_boxes, previous_boxes)
    previous_centers = (previous_boxes[:, :2] + previous_boxes[:, 2:]) * 0.5
    current_centers = (current_boxes[:, :2] + current_boxes[:, 2:]) * 0.5
    distance = np.linalg.norm(current_centers[:, None] - previous_centers[None, :], axis=-1)
    keep = min(top_k, len(previous))
    pairs = []
    for current_index in range(len(current)):
        by_iou = np.argsort(-overlap[current_index], kind="mergesort")[:keep]
        by_distance = np.argsort(distance[current_index], kind="mergesort")[:keep]
        pairs.extend((previous_index, current_index) for previous_index in sorted(set(by_iou.tolist() + by_distance.tolist())))
    return pairs


def pair_features(previous, current, pairs):
    if not pairs:
        return np.zeros((0, len(EDGE_NAMES)), dtype=np.float32)
    previous_boxes, current_boxes = tlwh_to_tlbr(previous[:, :4]), tlwh_to_tlbr(current[:, :4])
    all_iou = iou_matrix(current_boxes, previous_boxes)
    rows = []
    for previous_index, current_index in pairs:
        first, second = previous[previous_index], current[current_index]
        first_center = (previous_boxes[previous_index, :2] + previous_boxes[previous_index, 2:]) * 0.5
        second_center = (current_boxes[current_index, :2] + current_boxes[current_index, 2:]) * 0.5
        scale = max(math.sqrt(max(float(first[2] * first[3]), 1e-6)), 1e-3)
        rows.append((
            first[4], second[4], logit([first[4]])[0], logit([second[4]])[0],
            (second_center[0] - first_center[0]) / scale, (second_center[1] - first_center[1]) / scale,
            math.log(max(float(second[2]), 1e-3) / max(float(first[2]), 1e-3)),
            math.log(max(float(second[3]), 1e-3) / max(float(first[3]), 1e-3)),
            all_iou[current_index, previous_index], second[4] - first[4],
        ))
    return np.asarray(rows, dtype=np.float32)


def match_ids(current, gt_rows, threshold=0.5):
    ids = np.full(len(current), -1, dtype=np.int64)
    if not len(current) or not len(gt_rows):
        return (ids > 0).astype(np.float32), ids
    overlap = iou_matrix(tlwh_to_tlbr(current[:, :4]), tlwh_to_tlbr(gt_rows[:, 1:5]))
    cost = 1.0 - overlap
    cost[overlap < threshold] = 2.0
    rows, columns = linear_sum_assignment(cost)
    for current_index, gt_index in zip(rows, columns):
        if overlap[current_index, gt_index] >= threshold:
            ids[current_index] = int(gt_rows[gt_index, 0])
    return (ids > 0).astype(np.float32), ids


def extract_supervision(dataset_root, detections_root, sequences, split="train", top_k=3):
    nodes_x, nodes_y, edges_x, edges_y = [], [], [], []
    summary = {"split": split, "pair_top_k": top_k, "gt_iou_threshold": 0.5, "sequences": {}}
    for sequence in sequences:
        sequence_dir, det_path = dataset_root / split / sequence, detections_root / sequence / "det.txt"
        if not sequence_dir.is_dir() or not det_path.is_file():
            raise FileNotFoundError("missing sequence or detection cache for {}".format(sequence))
        info, detections, gt = read_seqinfo(sequence_dir / "seqinfo.ini"), read_detections(det_path), read_gt(sequence_dir / "gt" / "gt.txt")
        previous = np.zeros((0, 5), dtype=np.float32)
        previous_ids = np.zeros((0,), dtype=np.int64)
        sequence_nodes, sequence_edges = 0, 0
        for frame in range(1, info["frames"] + 1):
            current = detections.get(frame, np.zeros((0, 5), dtype=np.float32))
            current_node_y, current_ids = match_ids(current, gt.get(frame, np.zeros((0, 5), dtype=np.float32)))
            nodes_x.append(node_features(current, previous, info["width"], info["height"]))
            nodes_y.append(current_node_y)
            pairs = candidate_pairs(previous, current, top_k)
            if pairs:
                edges_x.append(pair_features(previous, current, pairs))
                edges_y.append(np.asarray([float(previous_ids[a] > 0 and previous_ids[a] == current_ids[b]) for a, b in pairs], dtype=np.float32))
                sequence_edges += len(pairs)
            sequence_nodes += len(current)
            previous, previous_ids = current, current_ids
        summary["sequences"][sequence] = {"frames": info["frames"], "node_samples": sequence_nodes, "edge_samples": sequence_edges, "det_sha256": sha256(det_path), "gt_sha256": sha256(sequence_dir / "gt" / "gt.txt")}
    data = {
        "node_x": np.concatenate(nodes_x).astype(np.float32), "node_y": np.concatenate(nodes_y).astype(np.float32),
        "edge_x": np.concatenate(edges_x).astype(np.float32) if edges_x else np.zeros((0, len(EDGE_NAMES)), dtype=np.float32),
        "edge_y": np.concatenate(edges_y).astype(np.float32) if edges_y else np.zeros((0,), dtype=np.float32),
    }
    summary["totals"] = {"node_samples": int(len(data["node_y"])), "node_positive": int(data["node_y"].sum()), "edge_samples": int(len(data["edge_y"])), "edge_positive": int(data["edge_y"].sum())}
    return data, summary


class MLP(nn.Module):
    def __init__(self, dimension, hidden_dim, dropout):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dimension, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, values):
        return self.network(values).squeeze(-1)


def scale_fit(values):
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(values.std(axis=0, dtype=np.float64).astype(np.float32), 1e-5)
    return mean, std


def scale(values, mean, std):
    return ((values - mean) / std).astype(np.float32)


def binary_auc(score, label):
    positive, negative = int((label > 0.5).sum()), int((label <= 0.5).sum())
    if not positive or not negative:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    rank, sorted_score, sorted_label = np.empty(len(score), dtype=np.float64), score[order], label[order]
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        rank[start:end] = (start + 1 + end) * 0.5
        start = end
    return float((rank[sorted_label > 0.5].sum() - positive * (positive + 1) * 0.5) / (positive * negative))


def predict_logits(model, values, device, batch_size=8192):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            outputs.append(model(torch.from_numpy(values[start:start + batch_size]).to(device)).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.zeros((0,), dtype=np.float32)


def metrics(model, values, labels, device):
    logits = predict_logits(model, values, device)
    probability = 1.0 / (1.0 + np.exp(-logits))
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return {"bce": float(-(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)).mean()), "auc": binary_auc(probability, labels), "positive_rate": float(labels.mean())}


def fit_models(fit, validation, output_dir, device_name, seed=0, epochs=40, hidden_dim=64, dropout=0.1, learning_rate=1e-3, batch_size=4096):
    if min(len(fit["node_y"]), len(fit["edge_y"]), len(validation["node_y"]), len(validation["edge_y"])) == 0:
        raise ValueError("node and edge supervision must be non-empty")
    torch.manual_seed(seed)
    generator, device = np.random.default_rng(seed), torch.device(device_name)
    node_mean, node_std, edge_mean, edge_std = *scale_fit(fit["node_x"]), *scale_fit(fit["edge_x"])
    node_fit, node_val = scale(fit["node_x"], node_mean, node_std), scale(validation["node_x"], node_mean, node_std)
    edge_fit, edge_val = scale(fit["edge_x"], edge_mean, edge_std), scale(validation["edge_x"], edge_mean, edge_std)
    node_model, edge_model = MLP(len(NODE_NAMES), hidden_dim, dropout).to(device), MLP(len(EDGE_NAMES), hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(list(node_model.parameters()) + list(edge_model.parameters()), lr=learning_rate, weight_decay=1e-4)
    pos_node = min((len(fit["node_y"]) - fit["node_y"].sum()) / max(fit["node_y"].sum(), 1.0), 20.0)
    pos_edge = min((len(fit["edge_y"]) - fit["edge_y"].sum()) / max(fit["edge_y"].sum(), 1.0), 20.0)
    node_loss, edge_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_node, device=device)), nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_edge, device=device))
    steps, history, best = max(math.ceil(len(node_fit) / batch_size), math.ceil(len(edge_fit) / batch_size)), [], {"score": float("inf"), "epoch": -1}
    checkpoint = output_dir / "checkpoint.pt"
    for epoch in range(1, epochs + 1):
        node_order, edge_order = generator.permutation(len(node_fit)), generator.permutation(len(edge_fit))
        node_model.train(); edge_model.train()
        for step in range(steps):
            node_indices = node_order[(step * batch_size) % len(node_fit):((step * batch_size) % len(node_fit)) + batch_size]
            edge_indices = edge_order[(step * batch_size) % len(edge_fit):((step * batch_size) % len(edge_fit)) + batch_size]
            if not len(node_indices): node_indices = node_order[:batch_size]
            if not len(edge_indices): edge_indices = edge_order[:batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = node_loss(node_model(torch.from_numpy(node_fit[node_indices]).to(device)), torch.from_numpy(fit["node_y"][node_indices]).to(device)) + edge_loss(edge_model(torch.from_numpy(edge_fit[edge_indices]).to(device)), torch.from_numpy(fit["edge_y"][edge_indices]).to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(list(node_model.parameters()) + list(edge_model.parameters()), 5.0); optimizer.step()
        node_metric, edge_metric = metrics(node_model, node_val, validation["node_y"], device), metrics(edge_model, edge_val, validation["edge_y"], device)
        score = node_metric["bce"] + edge_metric["bce"]
        history.append({"epoch": epoch, "node": node_metric, "edge": edge_metric, "validation_bce_sum": score})
        if score < best["score"]:
            best = {"score": score, "epoch": epoch}
            torch.save({"schema_version": 1, "method": "causal_low_score_lineage_rescorer", "node_names": NODE_NAMES, "edge_names": EDGE_NAMES, "hidden_dim": hidden_dim, "dropout": dropout, "node_state": node_model.state_dict(), "edge_state": edge_model.state_dict(), "node_mean": node_mean, "node_std": node_std, "edge_mean": edge_mean, "edge_std": edge_std, "selected_epoch": epoch, "seed": seed}, checkpoint)
    return {"checkpoint": str(checkpoint), "best": best, "epochs": history}


class Predictor:
    def __init__(self, checkpoint_path, device_name):
        self.device = torch.device(device_name)
        payload = torch.load(checkpoint_path, map_location=self.device)
        if payload.get("schema_version") != 1 or payload.get("method") != "causal_low_score_lineage_rescorer":
            raise ValueError("unsupported checkpoint: {}".format(checkpoint_path))
        self.node, self.edge = MLP(len(NODE_NAMES), payload["hidden_dim"], payload["dropout"]).to(self.device), MLP(len(EDGE_NAMES), payload["hidden_dim"], payload["dropout"]).to(self.device)
        self.node.load_state_dict(payload["node_state"]); self.edge.load_state_dict(payload["edge_state"])
        self.node.eval(); self.edge.eval()
        self.node_mean, self.node_std, self.edge_mean, self.edge_std = [np.asarray(payload[key], dtype=np.float32) for key in ("node_mean", "node_std", "edge_mean", "edge_std")]

    def node_probability(self, values):
        return (1.0 / (1.0 + np.exp(-predict_logits(self.node, scale(values, self.node_mean, self.node_std), self.device)))).astype(np.float32)

    def edge_probability(self, values):
        return (1.0 / (1.0 + np.exp(-predict_logits(self.edge, scale(values, self.edge_mean, self.edge_std), self.device)))).astype(np.float32)


def write_detections(path, rows):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as handle:
        temporary = Path(handle.name)
        for frame, x, y, width, height, score in rows:
            handle.write("{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.8f},-1,-1,-1\n".format(frame, -1, x, y, width, height, score))
    os.replace(str(temporary), str(path))


def rescore_cache(dataset_root, input_root, output_root, checkpoint_path, sequences, split, device_name, quality_threshold, link_threshold, history_minimum=2, promotion_floor=0.71, pair_top_k=3):
    if output_root.exists():
        raise FileExistsError("refusing to overwrite cache: {}".format(output_root))
    if not (0.0 < quality_threshold < 1.0 and 0.0 < link_threshold < 1.0 and 0.0 < promotion_floor < 1.0) or history_minimum < 2:
        raise ValueError("invalid promotion parameters")
    predictor, summary = Predictor(checkpoint_path, device_name), {"split": split, "inference_is_gt_free": True, "checkpoint_sha256": sha256(checkpoint_path), "sequences": {}}
    output_root.mkdir(parents=True)
    for sequence in sequences:
        sequence_dir, source_path = dataset_root / split / sequence, input_root / sequence / "det.txt"
        if not sequence_dir.is_dir() or not source_path.is_file():
            raise FileNotFoundError("missing sequence or detector cache for {}".format(sequence))
        info, detections = read_seqinfo(sequence_dir / "seqinfo.ini"), read_detections(source_path)
        previous, previous_quality, previous_history = np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        rows, promoted, total = [], 0, 0
        target_dir = output_root / sequence; target_dir.mkdir()
        for frame in range(1, info["frames"] + 1):
            current = detections.get(frame, np.zeros((0, 5), dtype=np.float32))
            quality, history, allow = predictor.node_probability(node_features(current, previous, info["width"], info["height"])), np.ones(len(current), dtype=np.int64), np.zeros(len(current), dtype=bool)
            pairs = candidate_pairs(previous, current, pair_top_k)
            if pairs:
                probabilities = predictor.edge_probability(pair_features(previous, current, pairs))
                best_probability, best_previous = np.full(len(current), -1.0, dtype=np.float32), np.full(len(current), -1, dtype=np.int64)
                for (previous_index, current_index), probability in zip(pairs, probabilities):
                    if probability > best_probability[current_index]: best_probability[current_index], best_previous[current_index] = probability, previous_index
                for index, predecessor in enumerate(best_previous):
                    if predecessor >= 0 and quality[index] >= quality_threshold and previous_quality[predecessor] >= quality_threshold and best_probability[index] >= link_threshold:
                        history[index] = previous_history[predecessor] + 1; allow[index] = history[index] >= history_minimum
            scores = current[:, 4].copy()
            calibrated = promotion_floor + (1.0 - promotion_floor) * np.clip((quality - quality_threshold) / max(1.0 - quality_threshold, 1e-6), 0.0, 1.0)
            promote = allow & (calibrated > scores)
            scores[promote] = calibrated[promote]
            promoted += int(promote.sum()); total += len(scores)
            rows.extend((frame, float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(score)) for row, score in zip(current, scores))
            previous, previous_quality, previous_history = current, quality, history
        target = target_dir / "det.txt"; write_detections(target, rows)
        summary["sequences"][sequence] = {"detections": total, "promotions": promoted, "source_sha256": sha256(source_path), "output_sha256": sha256(target)}
    summary["parameters"] = {"quality_threshold": quality_threshold, "link_threshold": link_threshold, "history_minimum": history_minimum, "promotion_floor": promotion_floor, "pair_top_k": pair_top_k}
    summary["totals"] = {"sequences": len(sequences), "detections": int(sum(item["detections"] for item in summary["sequences"].values())), "promotions": int(sum(item["promotions"] for item in summary["sequences"].values()))}
    summary["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    (output_root / "cache_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
