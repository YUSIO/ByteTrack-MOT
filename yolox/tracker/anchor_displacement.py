"""Deterministic anchor-displacement association for ByteTrack.

The module is deliberately restricted to ambiguous components of the first,
high-confidence association.  It uses only provisional baseline matches from
the current frame to estimate a shared displacement; it never reads ground
truth, creates pseudo detections, or updates track state before the final
Hungarian assignment.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .basetrack import TrackState


def box_center(tlwh) -> np.ndarray:
    """Return an ``(x, y)`` center in the tracker coordinate system."""

    box = np.asarray(tlwh, dtype=np.float64)
    return box[:2] + 0.5 * box[2:4]


def weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Coordinate-wise weighted median for a small deterministic point set."""

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
        raise ValueError("values must have shape (N, 2) with N > 0")
    if weights.shape != (len(values),) or not np.all(np.isfinite(weights)):
        raise ValueError("weights must have shape (N,) and be finite")
    weights = np.maximum(weights, 0.0)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(values), dtype=np.float64)

    result = np.empty(2, dtype=np.float64)
    for axis in range(2):
        order = np.argsort(values[:, axis], kind="mergesort")
        sorted_values = values[order, axis]
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights)
        result[axis] = sorted_values[np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left")]
    return result


def _candidate_components(cost_matrix: np.ndarray, threshold: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return connected components of the finite baseline candidate graph."""

    candidate = np.isfinite(cost_matrix) & (cost_matrix <= float(threshold))
    n_tracks, n_detections = candidate.shape
    visited_tracks = set()
    visited_detections = set()
    components = []

    for start_track in range(n_tracks):
        if start_track in visited_tracks or not candidate[start_track].any():
            continue
        tracks = set()
        detections = set()
        pending_tracks = [start_track]
        while pending_tracks:
            track_index = pending_tracks.pop()
            if track_index in visited_tracks:
                continue
            visited_tracks.add(track_index)
            tracks.add(track_index)
            for detection_index in np.flatnonzero(candidate[track_index]):
                detection_index = int(detection_index)
                if detection_index in visited_detections:
                    continue
                visited_detections.add(detection_index)
                detections.add(detection_index)
                linked_tracks = np.flatnonzero(candidate[:, detection_index])
                pending_tracks.extend(int(index) for index in linked_tracks if int(index) not in visited_tracks)

        # A one-edge component cannot contain an association ambiguity.
        if len(tracks) > 1 or len(detections) > 1:
            components.append((np.asarray(sorted(tracks), dtype=int), np.asarray(sorted(detections), dtype=int)))
    return components


def _track_is_anchor(track, frame_id: int, prior_centers: Mapping[int, np.ndarray], min_age: int) -> bool:
    if getattr(track, "state", None) != TrackState.Tracked:
        return False
    if int(getattr(track, "frame_id", -1)) != int(frame_id) - 1:
        return False
    if int(getattr(track, "tracklet_len", 0)) < int(min_age):
        return False
    return int(getattr(track, "track_id", -1)) in prior_centers


def _row_margin(cost_matrix: np.ndarray, row: int, column: int, threshold: float) -> float:
    candidates = np.asarray(cost_matrix[row], dtype=np.float64)
    candidates = np.sort(candidates[np.isfinite(candidates) & (candidates <= threshold)])
    if len(candidates) < 2:
        return float("inf")
    if not np.isclose(candidates[0], cost_matrix[row, column]):
        return 0.0
    return float(candidates[1] - candidates[0])


def _selected_anchor_indices(
    tracks,
    detections,
    base_cost: np.ndarray,
    base_matches: np.ndarray,
    component_tracks: np.ndarray,
    component_detections: np.ndarray,
    prior_centers: Mapping[int, np.ndarray],
    frame_id: int,
    *,
    anchor_cost_threshold: float,
    anchor_margin: float,
    anchor_min_age: int,
) -> Dict[int, int]:
    """Map safe baseline track indices to their matched detection indices."""

    component_track_set = set(int(index) for index in component_tracks)
    component_detection_set = set(int(index) for index in component_detections)
    selected = {}
    for pair in np.asarray(base_matches, dtype=int).reshape(-1, 2):
        track_index, detection_index = (int(pair[0]), int(pair[1]))
        if track_index in component_track_set or detection_index in component_detection_set:
            continue
        if base_cost[track_index, detection_index] > anchor_cost_threshold:
            continue
        if _row_margin(base_cost, track_index, detection_index, anchor_cost_threshold) < anchor_margin:
            continue
        if not _track_is_anchor(tracks[track_index], frame_id, prior_centers, anchor_min_age):
            continue
        if float(getattr(detections[detection_index], "score", 0.0)) <= 0.0:
            continue
        selected[track_index] = detection_index
    return selected


def _predict_from_anchors(
    target,
    anchors: Sequence[Tuple[object, object]],
    prior_centers: Mapping[int, np.ndarray],
    *,
    max_anchors: int,
    radius: float,
    sigma_floor: float,
    residual_threshold: float,
):
    target_id = int(getattr(target, "track_id", -1))
    if target_id not in prior_centers or len(anchors) < 1:
        return None
    target_center = np.asarray(prior_centers[target_id], dtype=np.float64)
    ranked = []
    for anchor_track, anchor_detection in anchors:
        anchor_id = int(getattr(anchor_track, "track_id", -1))
        if anchor_id not in prior_centers:
            continue
        anchor_previous = np.asarray(prior_centers[anchor_id], dtype=np.float64)
        anchor_current = box_center(anchor_detection.tlwh)
        displacement = anchor_current - anchor_previous
        distance = float(np.linalg.norm(anchor_previous - target_center))
        score = float(np.clip(getattr(anchor_detection, "score", 0.0), 0.0, 1.0))
        weight = max(score, 1e-3) * np.exp(-0.5 * (distance / max(radius, 1e-6)) ** 2)
        ranked.append((distance, anchor_previous, displacement, weight, anchor_id))
    ranked.sort(key=lambda item: (item[0], item[4]))
    ranked = ranked[: max(1, int(max_anchors))]
    if len(ranked) == 0:
        return None

    displacements = np.asarray([item[2] for item in ranked], dtype=np.float64)
    weights = np.asarray([item[3] for item in ranked], dtype=np.float64)
    displacement = weighted_median(displacements, weights)
    residuals = np.linalg.norm(displacements - displacement[None, :], axis=1)
    robust_scale = 1.4826 * float(np.median(np.abs(residuals - np.median(residuals))))
    sigma = max(float(sigma_floor), robust_scale, 1e-6)
    prediction = target_center + displacement
    return {
        "prediction": prediction,
        "displacement": displacement,
        "sigma": sigma,
        "residual": float(np.median(residuals)),
        "anchor_ids": [int(item[4]) for item in ranked],
        "anchor_count": len(ranked),
        "eligible": len(ranked) >= 1 and float(np.median(residuals)) <= residual_threshold,
    }


def associate_with_anchor_displacement(
    tracks,
    detections,
    base_cost: np.ndarray,
    base_matches: np.ndarray,
    prior_centers: Mapping[int, np.ndarray],
    frame_id: int,
    *,
    match_threshold: float,
    anchor_min_count: int = 3,
    anchor_max_count: int = 5,
    anchor_cost_threshold: float = 0.75,
    anchor_margin: float = 0.05,
    anchor_min_age: int = 3,
    anchor_radius: float = 200.0,
    anchor_sigma_floor: float = 4.0,
    anchor_residual_threshold: float = 25.0,
    anchor_lambda: float = 0.25,
):
    """Refine ambiguous first-stage assignments with deterministic anchors.

    The candidate graph is never expanded: edges rejected by ``base_cost``
    remain rejected.  The returned matches are globally one-to-one and the
    unmatched indices are recomputed from the final assignment.
    """

    base_cost = np.asarray(base_cost, dtype=np.float64)
    base_matches = np.asarray(base_matches, dtype=int).reshape(-1, 2)
    if base_cost.size == 0 or len(tracks) == 0 or len(detections) == 0:
        return base_matches, tuple(range(len(tracks))), tuple(range(len(detections))), []

    components = _candidate_components(base_cost, match_threshold)
    if not components:
        matched_tracks = set(int(pair[0]) for pair in base_matches)
        matched_detections = set(int(pair[1]) for pair in base_matches)
        return (
            base_matches,
            tuple(index for index in range(len(tracks)) if index not in matched_tracks),
            tuple(index for index in range(len(detections)) if index not in matched_detections),
            [],
        )

    final_pairs = []
    traces = []
    component_track_indices = set()
    component_detection_indices = set()
    base_match_by_track = {int(pair[0]): int(pair[1]) for pair in base_matches}

    for component_id, (track_indices, detection_indices) in enumerate(components):
        component_track_indices.update(int(index) for index in track_indices)
        component_detection_indices.update(int(index) for index in detection_indices)
        anchor_matches = _selected_anchor_indices(
            tracks,
            detections,
            base_cost,
            base_matches,
            track_indices,
            detection_indices,
            prior_centers,
            frame_id,
            anchor_cost_threshold=anchor_cost_threshold,
            anchor_margin=anchor_margin,
            anchor_min_age=anchor_min_age,
        )
        anchor_pairs = [
            (tracks[track_index], detections[detection_index])
            for track_index, detection_index in sorted(anchor_matches.items())
        ]
        refined = base_cost[np.ix_(track_indices, detection_indices)].copy()
        target_details = []

        for local_track_index, track_index in enumerate(track_indices):
            target = tracks[int(track_index)]
            prediction = _predict_from_anchors(
                target,
                anchor_pairs,
                prior_centers,
                max_anchors=anchor_max_count,
                radius=anchor_radius,
                sigma_floor=anchor_sigma_floor,
                residual_threshold=anchor_residual_threshold,
            )
            if prediction is None or prediction["anchor_count"] < int(anchor_min_count) or not prediction["eligible"]:
                continue
            candidate_centers = np.asarray([box_center(detections[int(index)].tlwh) for index in detection_indices])
            candidate_distance = np.linalg.norm(candidate_centers - prediction["prediction"][None, :], axis=1)
            anchor_cost = np.clip((candidate_distance / prediction["sigma"]) ** 2 / 2.0, 0.0, 1.0)
            valid = np.isfinite(refined[local_track_index]) & (base_cost[track_index, detection_indices] <= match_threshold)
            refined[local_track_index, valid] = (
                (1.0 - float(anchor_lambda)) * refined[local_track_index, valid]
                + float(anchor_lambda) * anchor_cost[valid]
            )
            target_details.append(
                {
                    "track_index": int(track_index),
                    "track_id": int(getattr(target, "track_id", -1)),
                    "prediction": prediction["prediction"].tolist(),
                    "displacement": prediction["displacement"].tolist(),
                    "sigma": float(prediction["sigma"]),
                    "residual": float(prediction["residual"]),
                    "anchor_ids": prediction["anchor_ids"],
                }
            )

        local_matches, _, _ = _linear_assignment(refined, match_threshold)
        for local_track, local_detection in local_matches:
            final_pairs.append((int(track_indices[local_track]), int(detection_indices[local_detection])))
        traces.append(
            {
                "component_id": int(component_id),
                "track_indices": [int(index) for index in track_indices],
                "detection_indices": [int(index) for index in detection_indices],
                "anchor_track_ids": [
                    int(getattr(anchor_track, "track_id", -1))
                    for anchor_track, _ in anchor_pairs
                ],
                "base_pairs": [
                    [int(track_index), int(detection_index)]
                    for track_index, detection_index in base_match_by_track.items()
                    if track_index in set(int(index) for index in track_indices)
                ],
                "final_pairs": [
                    [int(track_index), int(detection_index)]
                    for track_index, detection_index in final_pairs
                    if track_index in set(int(index) for index in track_indices)
                ],
                "targets": target_details,
            }
        )

    # Components are disjoint.  Preserve the provisional baseline matches
    # outside them and use the refined assignment inside each component.
    for track_index, detection_index in base_matches:
        track_index = int(track_index)
        detection_index = int(detection_index)
        if track_index not in component_track_indices and detection_index not in component_detection_indices:
            final_pairs.append((track_index, detection_index))

    final_pairs = np.asarray(sorted(set(final_pairs)), dtype=int)
    if final_pairs.size == 0:
        final_pairs = np.empty((0, 2), dtype=int)
    else:
        final_pairs = final_pairs.reshape(-1, 2)
    matched_tracks = set(int(pair[0]) for pair in final_pairs)
    matched_detections = set(int(pair[1]) for pair in final_pairs)
    unmatched_tracks = tuple(index for index in range(len(tracks)) if index not in matched_tracks)
    unmatched_detections = tuple(index for index in range(len(detections)) if index not in matched_detections)
    return final_pairs, unmatched_tracks, unmatched_detections, traces


def _linear_assignment(cost_matrix: np.ndarray, threshold: float):
    """Small local Hungarian wrapper using SciPy's deterministic solver."""

    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    finite = np.isfinite(cost_matrix)
    safe_cost = np.where(finite, cost_matrix, 1e6)
    rows, columns = linear_sum_assignment(safe_cost)
    keep = safe_cost[rows, columns] <= float(threshold)
    pairs = np.column_stack((rows[keep], columns[keep])).astype(int)
    unmatched_rows = tuple(index for index in range(cost_matrix.shape[0]) if index not in set(pairs[:, 0]))
    unmatched_columns = tuple(index for index in range(cost_matrix.shape[1]) if index not in set(pairs[:, 1]))
    return pairs, unmatched_rows, unmatched_columns
