"""Detection-to-track association logic.

Implements single-stage and BYTE two-stage association. The first stage
matches high-confidence detections to predicted tracks using IoU + OC-SORT's
observation-centric momentum (OCM) direction consistency. The second stage
(BYTE) matches remaining unmatched tracks against low-confidence detections
using IoU only.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from arcsort.matching import (
    eiou_batch,
    embedding_similarity_batch,
    hybrid_score_batch,
    iou_batch,
    linear_assignment,
)
from arcsort.ocsort_components import compute_ocm_cost


def _compute_adaptive_weight(
    emb_cost: np.ndarray,
    w_assoc_emb: float,
    bottom: float = 0.5,
) -> np.ndarray:
    """Compute per-element adaptive weight for embedding cost.

    For each detection (row) and each track (column), the weight is scaled
    down when the best and second-best embedding matches are close in score,
    indicating ambiguity. This prevents the embedding cost from dominating
    when appearance is not discriminative.

    Follows the Deep-OC-SORT ``compute_aw_max_metric`` pattern.

    Args:
        emb_cost: Raw embedding similarity matrix, shape (N, M).
        w_assoc_emb: Base embedding weight.
        bottom: Ratio threshold below which weight is full. Range [0, 1].

    Returns:
        Weighted embedding cost matrix, shape (N, M).
    """
    w_emb = np.full_like(emb_cost, w_assoc_emb)

    # When bottom >= 1.0, adaptive weighting is effectively disabled
    # (all ratios are below bottom, so weight stays at full)
    if bottom >= 1.0:
        return w_emb * emb_cost

    def _ambiguity_weights(best: np.ndarray, second: np.ndarray) -> np.ndarray:
        safe_best = np.where(best == 0, 1.0, best)
        ratio = second / safe_best
        weights = 1.0 - np.maximum(ratio - bottom, 0.0) / (1.0 - bottom)
        return np.where(best == 0, 0.0, weights)

    # Row-wise: per-detection ambiguity (needs >= 2 candidate tracks)
    if emb_cost.shape[1] >= 2:
        top2 = np.partition(emb_cost, -2, axis=1)
        w_emb *= _ambiguity_weights(top2[:, -1], top2[:, -2])[:, np.newaxis]

    # Column-wise: per-track ambiguity (needs >= 2 candidate detections)
    if emb_cost.shape[0] >= 2:
        top2 = np.partition(emb_cost, -2, axis=0)
        w_emb *= _ambiguity_weights(top2[-1], top2[-2])[np.newaxis, :]

    return w_emb * emb_cost


def velocity_continuity_cost(
    detections: np.ndarray,
    track_predictions: np.ndarray,
    track_filter_vels: np.ndarray,
    track_last_obs: np.ndarray,
    track_dt: np.ndarray,
    weight: float,
    r_cap: float,
    max_dt: int = 2,
) -> np.ndarray:
    """Signed velocity-continuity bonus for the stage-1 cost matrix.

    At a crossing, the competing boxes nearly coincide (spatial scores
    tie, appearance is occluded or costume-degenerate) but the targets
    keep their velocities — a swapped assignment implies a velocity
    discontinuity. For each pair, compare the velocity implied by
    matching detection i to track j, (c_det − c_last_obs)/Δt, with the
    filter's velocity estimate; the residual r (normalized by predicted
    box extent, so units are box-widths per frame) maps to a signed
    bonus weight·(1/2 − min(r/r_cap, 1)) ∈ [−w/2, +w/2], the same shape
    as OCM's inertia term. Tracks unseen for more than max_dt frames
    contribute 0 — a coasted velocity compared against a many-frame
    displacement average is noise, not evidence.

    Args:
        detections: (N, 4) xyxy.
        track_predictions: (M, 4) xyxy predicted (post-CMC) boxes.
        track_filter_vels: (M, 2) filter center velocities; NaN rows
            disable the cue for that track.
        track_last_obs: (M, 4+) xyxy last observed boxes.
        track_dt: (M,) frames since last observation (>= 1).
        weight: Bonus magnitude (like OCM's inertia_weight).
        r_cap: Residual (in box-widths/frame) at which the bonus
            saturates at −weight/2.
        max_dt: Tracks with dt > max_dt get a zero term.

    Returns:
        (N, M) additive score term.
    """
    n, m = detections.shape[0], track_predictions.shape[0]
    if n == 0 or m == 0 or weight == 0:
        return np.zeros((n, m))

    det_c = (detections[:, :2] + detections[:, 2:4]) / 2          # (N, 2)
    obs_c = (track_last_obs[:, :2] + track_last_obs[:, 2:4]) / 2  # (M, 2)
    dt = np.maximum(track_dt.astype(np.float64), 1.0)             # (M,)

    v_implied = (det_c[:, None, :] - obs_c[None, :, :]) / dt[None, :, None]
    resid = np.linalg.norm(v_implied - track_filter_vels[None, :, :], axis=2)

    extent = np.maximum(
        (track_predictions[:, 2] - track_predictions[:, 0])
        + (track_predictions[:, 3] - track_predictions[:, 1]),
        2.0,
    ) / 2.0                                                        # (M,)
    r = resid / extent[None, :]

    bonus = weight * (0.5 - np.minimum(r / max(r_cap, 1e-6), 1.0))

    valid = (track_dt <= max_dt) & ~np.isnan(track_filter_vels).any(axis=1)
    bonus[:, ~valid] = 0.0
    return bonus


def associate(
    detections: np.ndarray,
    track_predictions: np.ndarray,
    track_velocities: np.ndarray,
    track_previous_obs: np.ndarray,
    iou_threshold: float,
    inertia_weight: float,
    use_hybrid_distance: bool = False,
    hybrid_config: dict | None = None,
    emb_cost: np.ndarray | None = None,
    w_assoc_emb: float = 0.75,
    aw_off: bool = False,
    aw_param: float = 0.5,
    gate_mask: np.ndarray | None = None,
    emb_rescue_thresh: float | None = None,
    track_expansions: np.ndarray | None = None,
    spatial_score: np.ndarray | None = None,
    return_diagnostics: bool = False,
    vel_cue: dict | None = None,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, dict]
):
    """Associate detections to tracks using IoU/hybrid + OCM + ReID.

    Builds a cost matrix combining spatial similarity, direction
    consistency, and (optionally) appearance embedding similarity.
    Solves the linear assignment problem and filters matches below
    the score threshold.

    When emb_cost is provided, the final score is:
        ``score = score_base + ocm + w * embedding_similarity``
    where w is either a fixed weight (aw_off=True) or adaptively
    computed per element (aw_off=False) based on match ambiguity.

    Embedding cost is masked to zero wherever the base spatial score
    is <= 0 to prevent pure-appearance matches with no spatial support.

    Args:
        detections: Detection bboxes, shape (N, 4) in [x1, y1, x2, y2].
        track_predictions: Predicted track bboxes, shape (M, 4).
        track_velocities: Per-track velocity directions, shape (M, 2).
        track_previous_obs: Per-track previous observations, shape (M, 5).
        iou_threshold: Minimum score for a valid match.
        inertia_weight: Weight for OCM direction consistency cost.
        use_hybrid_distance: If True, use hybrid IoU/center fallback.
        hybrid_config: Config dict for hybrid distance params.
        emb_cost: Embedding similarity matrix, shape (N, M). Higher =
            more similar. None to disable ReID fusion.
        w_assoc_emb: Base weight for embedding cost. Default 0.75.
        aw_off: If True, use fixed weight. If False, use adaptive
            weighting based on 1st/2nd best ratio. Default False.
        aw_param: Bottom threshold for adaptive weighting. Default 0.5.
        gate_mask: Boolean matrix, shape (N, M). False forbids the
            pair regardless of its score (e.g. Mahalanobis motion
            gate). None to disable gating.
        emb_rescue_thresh: Minimum embedding similarity for appearance
            to rescue a zero-overlap pair. Rescue is only permitted
            inside the motion gate, so both emb_cost and gate_mask must
            be provided; otherwise no rescue happens. None disables
            rescue (appearance can then only re-rank overlapping pairs).
        track_expansions: Per-track ExpansionIoU factors, shape (M,),
            or (M, 2) with anisotropic [ex, ey] columns. When provided,
            the spatial score is EIoU with these factors (overrides
            use_hybrid_distance). None disables EIoU.
        return_diagnostics: If True, return a fourth element — a dict
            with the assembled score matrices ("score": the full matrix
            the solver saw, "spatial": the base spatial score) for
            downstream ambiguity analysis. Never alters the assignment.
        vel_cue: Optional velocity-continuity cue inputs (see
            velocity_continuity_cost): {"filter_vels": (M, 2),
            "last_obs": (M, 4+), "dt": (M,), "weight": float,
            "r_cap": float}. None disables the term.
        spatial_score: Precomputed spatial score matrix, shape (N, M)
            (e.g. sigma-DIoU for eiou.mode "sigma"). Overrides
            track_expansions and use_hybrid_distance. May contain
            negative values (DIoU range (-1, 1]); such pairs can never
            become matches because acceptance requires
            ``score_base >= iou_threshold`` with a positive threshold.
            The score is deliberately NOT clamped at 0: the negative
            range preserves the assignment-ranking gradient between
            separated pairs, and negative-score assignments are
            filtered out by the acceptance test below.

    Returns:
        Tuple of:
            matched: Matched pairs as array of shape (K, 2),
                columns are [detection_idx, track_idx].
            unmatched_dets: Indices of unmatched detections, shape (L,).
            unmatched_trks: Indices of unmatched tracks, shape (P,).
    """
    num_dets = detections.shape[0]
    num_trks = track_predictions.shape[0]

    if num_trks == 0 or num_dets == 0:
        empty = (
            np.empty((0, 2), dtype=int),
            np.arange(num_dets, dtype=int) if num_trks == 0
            else np.empty(0, dtype=int),
            np.empty(0, dtype=int) if num_trks == 0
            else np.arange(num_trks, dtype=int),
        )
        if return_diagnostics:
            zeros = np.zeros((num_dets, num_trks))
            return (*empty, {"score": zeros, "spatial": zeros})
        return empty

    # Similarity score: precomputed (sigma-DIoU), EIoU, hybrid, or IoU
    if spatial_score is not None:
        score_base = spatial_score
    elif track_expansions is not None:
        score_base = eiou_batch(detections, track_predictions, track_expansions)
    elif use_hybrid_distance:
        hcfg = hybrid_config or {}
        score_base = hybrid_score_batch(
            detections,
            track_predictions,
            iou_fallback_thresh=hcfg.get("iou_fallback_thresh", 0.1),
            shape_penalty_weight=hcfg.get("shape_penalty_weight", 0.3),
            fallback_cap=hcfg.get("fallback_cap", 0.5),
        )
    else:
        score_base = iou_batch(detections, track_predictions)

    # OCM direction consistency cost
    ocm_cost = compute_ocm_cost(
        detections, track_velocities, track_previous_obs, inertia_weight
    )

    # Appearance rescue: pairs inside the motion gate whose embedding
    # similarity is high enough may match without spatial overlap.
    rescue_eligible = None
    if (
        emb_rescue_thresh is not None
        and emb_cost is not None
        and gate_mask is not None
    ):
        rescue_eligible = gate_mask & (emb_cost >= emb_rescue_thresh)

    # Embedding appearance cost (additive fusion)
    appearance_cost = 0
    if emb_cost is not None:
        # Mask embedding cost where spatial score is zero (no overlap),
        # except for rescue-eligible pairs which keep their appearance
        emb_masked = emb_cost.copy()
        no_overlap = score_base <= 0
        if rescue_eligible is not None:
            no_overlap = no_overlap & ~rescue_eligible
        emb_masked[no_overlap] = 0

        if not aw_off:
            appearance_cost = _compute_adaptive_weight(
                emb_masked, w_assoc_emb, bottom=aw_param
            )
        else:
            appearance_cost = w_assoc_emb * emb_masked

    # Velocity-continuity bonus (crossing tie-breaker)
    vel_cost = 0
    if vel_cue is not None:
        vel_cost = velocity_continuity_cost(
            detections, track_predictions,
            vel_cue["filter_vels"], vel_cue["last_obs"], vel_cue["dt"],
            weight=vel_cue["weight"], r_cap=vel_cue["r_cap"],
        )

    # Combined score to maximize
    score_matrix = score_base + ocm_cost + appearance_cost + vel_cost

    # Forbid pairs outside the motion gate so the solver routes around them
    if gate_mask is not None:
        score_matrix = np.where(gate_mask, score_matrix, -1e6)

    # Solve assignment (minimize negative score)
    matched_indices = linear_assignment(-score_matrix)

    # Separate matched and unmatched
    matched_det_indices = set(matched_indices[:, 0]) if matched_indices.size > 0 else set()
    matched_trk_indices = set(matched_indices[:, 1]) if matched_indices.size > 0 else set()

    unmatched_dets = np.array(
        [d for d in range(num_dets) if d not in matched_det_indices], dtype=int
    )
    unmatched_trks = np.array(
        [t for t in range(num_trks) if t not in matched_trk_indices], dtype=int
    )

    # Filter matches below score threshold or outside the motion gate.
    # Rescue-eligible pairs are accepted on appearance despite low overlap.
    valid_matches = []
    for det_idx, trk_idx in matched_indices:
        gated_out = gate_mask is not None and not gate_mask[det_idx, trk_idx]
        rescued = (
            rescue_eligible is not None and rescue_eligible[det_idx, trk_idx]
        )
        spatially_valid = score_base[det_idx, trk_idx] >= iou_threshold
        if gated_out or not (spatially_valid or rescued):
            unmatched_dets = np.append(unmatched_dets, det_idx)
            unmatched_trks = np.append(unmatched_trks, trk_idx)
        else:
            valid_matches.append([det_idx, trk_idx])

    if len(valid_matches) == 0:
        matched = np.empty((0, 2), dtype=int)
    else:
        matched = np.array(valid_matches, dtype=int)

    if return_diagnostics:
        return matched, unmatched_dets, unmatched_trks, {
            "score": score_matrix,
            "spatial": score_base,
        }
    return matched, unmatched_dets, unmatched_trks


def swap_margins(
    matched: np.ndarray,
    score: np.ndarray,
    spatial: np.ndarray,
) -> list[tuple[int, int, float]]:
    """Permutation margins between accepted stage-1 matches.

    For accepted matches a = (d_i, t_j) and b = (d_k, t_l), the margin
    Δ = S[i,j] + S[k,l] − S[i,l] − S[k,j] measures how much total score
    the assignment loses if the two identities are swapped. A small Δ
    means the association decision was within noise of its permutation —
    the only reliable online witness of a potential identity swap. Pairs
    are only comparable when both cross-assignments have spatial support
    (spatial > 0); OCM's signed bonus alone must not create phantom
    swap candidates between distant targets.

    Returns [(match_idx_a, match_idx_b, margin)] for comparable pairs.
    """
    k = matched.shape[0]
    if k < 2:
        return []
    di = matched[:, 0]
    tj = matched[:, 1]
    cross = score[np.ix_(di, tj)]          # cross[a, b] = S(d_a, t_b)
    cross_sp = spatial[np.ix_(di, tj)]
    own = np.diag(cross)
    margin = own[:, None] + own[None, :] - cross - cross.T
    feasible = (cross_sp > 0) & (cross_sp.T > 0)
    out = []
    for a in range(k):
        for b in range(a + 1, k):
            if feasible[a, b]:
                out.append((a, b, float(margin[a, b])))
    return out


def associate_ocr(
    unmatched_detections: np.ndarray,
    unmatched_track_last_obs: np.ndarray,
    iou_threshold: float,
    emb_cost: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OCR: attempt recovery matching using last observations.

    Matches unmatched detections against tracks' last observed positions
    instead of predicted positions. This helps recover from filter
    divergence when tracks undergo sudden motion changes.

    When emb_cost is provided, the cost combines IoU and embedding
    similarity: ``cost = -(iou + emb_cost)``.

    Args:
        unmatched_detections: Unmatched detection bboxes, shape (N, 4).
        unmatched_track_last_obs: Last observed bboxes for unmatched
            tracks, shape (M, 4) in [x1, y1, x2, y2].
        iou_threshold: Minimum IoU for a valid recovery match.
        emb_cost: Embedding similarity matrix, shape (N, M). None to
            disable appearance in OCR.

    Returns:
        Tuple of:
            matched: Matched pairs, shape (K, 2).
            still_unmatched_dets: Remaining unmatched detection indices.
            still_unmatched_trks: Remaining unmatched track indices.
    """
    num_dets = unmatched_detections.shape[0]
    num_trks = unmatched_track_last_obs.shape[0]

    if num_dets == 0 or num_trks == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(num_dets, dtype=int),
            np.arange(num_trks, dtype=int),
        )

    iou_matrix = iou_batch(unmatched_detections, unmatched_track_last_obs)

    # Add embedding cost if available
    score_matrix = iou_matrix
    if emb_cost is not None:
        score_matrix = iou_matrix + emb_cost

    matched_indices = linear_assignment(-score_matrix)

    matched_det_indices = set(matched_indices[:, 0]) if matched_indices.size > 0 else set()
    matched_trk_indices = set(matched_indices[:, 1]) if matched_indices.size > 0 else set()

    still_unmatched_dets = np.array(
        [d for d in range(num_dets) if d not in matched_det_indices], dtype=int
    )
    still_unmatched_trks = np.array(
        [t for t in range(num_trks) if t not in matched_trk_indices], dtype=int
    )

    valid_matches = []
    for det_idx, trk_idx in matched_indices:
        if iou_matrix[det_idx, trk_idx] < iou_threshold:
            still_unmatched_dets = np.append(still_unmatched_dets, det_idx)
            still_unmatched_trks = np.append(still_unmatched_trks, trk_idx)
        else:
            valid_matches.append([det_idx, trk_idx])

    if len(valid_matches) == 0:
        matched = np.empty((0, 2), dtype=int)
    else:
        matched = np.array(valid_matches, dtype=int)

    return matched, still_unmatched_dets, still_unmatched_trks


def associate_byte_second(
    detections_low: np.ndarray,
    track_predictions: np.ndarray,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BYTE second-stage: match low-confidence detections to unmatched tracks.

    Uses IoU only (no OCM, no appearance features) since low-confidence
    detections have unreliable scores and crops. This conservative matching
    helps recover occluded objects that produce weak detections.

    Args:
        detections_low: Low-confidence detection bboxes, shape (N, 4)
            in [x1, y1, x2, y2].
        track_predictions: Predicted bboxes for unmatched tracks,
            shape (M, 4) in [x1, y1, x2, y2].
        iou_threshold: Minimum IoU for a valid match. Typically lower
            than first-stage threshold (e.g. 0.5).

    Returns:
        Tuple of:
            matched: Matched pairs, shape (K, 2) as
                [detection_idx, track_idx].
            unmatched_dets: Unmatched low-conf detection indices.
            unmatched_trks: Unmatched track indices.
    """
    num_dets = detections_low.shape[0]
    num_trks = track_predictions.shape[0]

    if num_dets == 0 or num_trks == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(num_dets, dtype=int),
            np.arange(num_trks, dtype=int),
        )

    iou_matrix = iou_batch(detections_low, track_predictions)

    matched_indices = linear_assignment(-iou_matrix)

    matched_det_indices = (
        set(matched_indices[:, 0]) if matched_indices.size > 0 else set()
    )
    matched_trk_indices = (
        set(matched_indices[:, 1]) if matched_indices.size > 0 else set()
    )

    unmatched_dets = np.array(
        [d for d in range(num_dets) if d not in matched_det_indices], dtype=int
    )
    unmatched_trks = np.array(
        [t for t in range(num_trks) if t not in matched_trk_indices], dtype=int
    )

    valid_matches = []
    for det_idx, trk_idx in matched_indices:
        if iou_matrix[det_idx, trk_idx] < iou_threshold:
            unmatched_dets = np.append(unmatched_dets, det_idx)
            unmatched_trks = np.append(unmatched_trks, trk_idx)
        else:
            valid_matches.append([det_idx, trk_idx])

    if len(valid_matches) == 0:
        matched = np.empty((0, 2), dtype=int)
    else:
        matched = np.array(valid_matches, dtype=int)

    return matched, unmatched_dets, unmatched_trks
