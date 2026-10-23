"""Distance computations and assignment solving for multi-object tracking.

Provides vectorized IoU computation, the hybrid IoU/center-distance metric,
and a wrapper around the Hungarian algorithm (via lap) for optimal assignment.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

try:
    import lap
    _USE_LAP = True
except ImportError:
    from scipy.optimize import linear_sum_assignment as scipy_lsa
    _USE_LAP = False
    logger.warning("lap not available, falling back to scipy.optimize.linear_sum_assignment")


def iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of bounding boxes.

    Fully vectorized implementation using broadcasting. No loops.

    Args:
        bboxes_a: Bounding boxes, shape (N, 4) in [x1, y1, x2, y2] format.
        bboxes_b: Bounding boxes, shape (M, 4) in [x1, y1, x2, y2] format.

    Returns:
        IoU matrix of shape (N, M) where element [i, j] is
        IoU(bboxes_a[i], bboxes_b[j]).
    """
    if bboxes_a.size == 0 or bboxes_b.size == 0:
        return np.empty((bboxes_a.shape[0], bboxes_b.shape[0]))

    # Expand dims for broadcasting: (N, 1, 4) vs (1, M, 4)
    a = bboxes_a[:, np.newaxis, :]
    b = bboxes_b[np.newaxis, :, :]

    # Intersection coordinates
    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)

    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])

    union = area_a[:, np.newaxis] + area_b[np.newaxis, :] - inter_area

    return inter_area / (union + 1e-6)


def hybrid_score_batch(
    bboxes_a: np.ndarray,
    bboxes_b: np.ndarray,
    iou_fallback_thresh: float = 0.1,
    shape_penalty_weight: float = 0.3,
    fallback_cap: float = 0.5,
) -> np.ndarray:
    """Compute pairwise hybrid similarity score between two sets of bboxes.

    Uses IoU when overlap exists. When IoU falls below the fallback
    threshold, switches to a center-distance + shape-penalty metric
    that can still produce meaningful scores for non-overlapping boxes.

    The output is a similarity score (higher = better match), comparable
    in scale to IoU so it can be added to OCM cost and solved by the
    same assignment framework.

    Args:
        bboxes_a: Bounding boxes, shape (N, 4) in [x1, y1, x2, y2].
        bboxes_b: Bounding boxes, shape (M, 4) in [x1, y1, x2, y2].
        iou_fallback_thresh: Below this IoU, use center distance fallback.
        shape_penalty_weight: Weight for the shape penalty term in fallback.
        fallback_cap: Maximum score a fallback (non-overlapping) pair can
            reach. CAUTION: if this is >= the association match threshold
            and no motion gate is active, non-overlapping boxes can match
            in the first stage — an ID-switch risk in crowded scenes. Set
            below the match threshold (or enable the Mahalanobis gate) to
            make fallback matches motion-bounded.

    Returns:
        Hybrid score matrix of shape (N, M). Higher = better match.
        Range approximately [0, 1].
    """
    if bboxes_a.size == 0 or bboxes_b.size == 0:
        return np.empty((bboxes_a.shape[0], bboxes_b.shape[0]))

    iou = iou_batch(bboxes_a, bboxes_b)

    # Compute center distance (normalized) for all pairs
    cx_a = (bboxes_a[:, 0] + bboxes_a[:, 2]) / 2
    cy_a = (bboxes_a[:, 1] + bboxes_a[:, 3]) / 2
    cx_b = (bboxes_b[:, 0] + bboxes_b[:, 2]) / 2
    cy_b = (bboxes_b[:, 1] + bboxes_b[:, 3]) / 2

    # Pairwise center distance: (N, M)
    dx = cx_a[:, np.newaxis] - cx_b[np.newaxis, :]
    dy = cy_a[:, np.newaxis] - cy_b[np.newaxis, :]
    center_dist = np.sqrt(dx ** 2 + dy ** 2)

    # Normalization: use the average diagonal of the two boxes in each pair
    w_a = bboxes_a[:, 2] - bboxes_a[:, 0]
    h_a = bboxes_a[:, 3] - bboxes_a[:, 1]
    w_b = bboxes_b[:, 2] - bboxes_b[:, 0]
    h_b = bboxes_b[:, 3] - bboxes_b[:, 1]

    diag_a = np.sqrt(w_a ** 2 + h_a ** 2)
    diag_b = np.sqrt(w_b ** 2 + h_b ** 2)
    avg_diag = (diag_a[:, np.newaxis] + diag_b[np.newaxis, :]) / 2
    norm_center_dist = center_dist / (avg_diag + 1e-6)

    # Shape penalty: |log(w1/w2)| + |log(h1/h2)|
    log_w_ratio = np.abs(
        np.log(w_a[:, np.newaxis] / (w_b[np.newaxis, :] + 1e-6) + 1e-6)
    )
    log_h_ratio = np.abs(
        np.log(h_a[:, np.newaxis] / (h_b[np.newaxis, :] + 1e-6) + 1e-6)
    )
    shape_penalty = (log_w_ratio + log_h_ratio) / 4.0

    # Fallback score: convert distance to similarity
    # Scale down so fallback scores are weaker than IoU matches. A fallback
    # score of 0.3 means "close center, similar shape" but is much weaker
    # evidence than IoU=0.3. The scaling keeps fallback matches conservative
    # to avoid false associations in crowded scenes.
    fallback_cost = norm_center_dist + shape_penalty_weight * shape_penalty
    fallback_score = np.maximum(1.0 - fallback_cost, 0.0)
    # Scale to [0, fallback_cap]: strong enough to pass a low threshold,
    # but weaker than any real IoU overlap
    fallback_score *= fallback_cap

    # Use IoU where it's above threshold, fallback where it's below
    use_fallback = iou < iou_fallback_thresh
    result = np.where(use_fallback, fallback_score, iou)

    return result


def eiou_batch(
    bboxes_a: np.ndarray,
    bboxes_b: np.ndarray,
    expansions_b: np.ndarray,
) -> np.ndarray:
    """Compute pairwise Expansion IoU with a per-column expansion factor.

    Both boxes of a pair are symmetrically expanded before the IoU is
    computed (Deep-EIoU, Huang et al., WACV-W 2024): each side of a box
    grows by ``e * dim``, so a box's width becomes ``w * (1 + 2e)``.
    Unlike Deep-EIoU's global constant, the expansion factor here is
    per-track (per column), which enables uncertainty-adaptive expansion:
    tracks with confident motion predictions keep tight boxes while
    maneuvering or occluded tracks search wider.

    Args:
        bboxes_a: Detection boxes, shape (N, 4) in [x1, y1, x2, y2].
        bboxes_b: Track boxes, shape (M, 4) in [x1, y1, x2, y2].
        expansions_b: Per-track expansion factors e >= 0. Shape (M,)
            for isotropic expansion, or (M, 2) with [ex, ey] columns
            for anisotropic expansion: each side grows by ``ex * w``
            horizontally and ``ey * h`` vertically.

    Returns:
        EIoU matrix of shape (N, M). With e=0 this equals iou_batch.
    """
    if bboxes_a.size == 0 or bboxes_b.size == 0:
        return np.empty((bboxes_a.shape[0], bboxes_b.shape[0]))

    e = np.asarray(expansions_b, dtype=np.float64)
    if e.ndim == 2:
        ex = e[:, 0][np.newaxis, :]  # (1, M)
        ey = e[:, 1][np.newaxis, :]
    else:
        ex = e[np.newaxis, :]  # (1, M)
        ey = ex

    aw = (bboxes_a[:, 2] - bboxes_a[:, 0])[:, np.newaxis]  # (N, 1)
    ah = (bboxes_a[:, 3] - bboxes_a[:, 1])[:, np.newaxis]
    ax1 = bboxes_a[:, 0][:, np.newaxis] - ex * aw
    ay1 = bboxes_a[:, 1][:, np.newaxis] - ey * ah
    ax2 = bboxes_a[:, 2][:, np.newaxis] + ex * aw
    ay2 = bboxes_a[:, 3][:, np.newaxis] + ey * ah

    bw = (bboxes_b[:, 2] - bboxes_b[:, 0])[np.newaxis, :]  # (1, M)
    bh = (bboxes_b[:, 3] - bboxes_b[:, 1])[np.newaxis, :]
    bx1 = bboxes_b[:, 0][np.newaxis, :] - ex * bw
    by1 = bboxes_b[:, 1][np.newaxis, :] - ey * bh
    bx2 = bboxes_b[:, 2][np.newaxis, :] + ex * bw
    by2 = bboxes_b[:, 3][np.newaxis, :] + ey * bh

    inter_w = np.maximum(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0)
    inter_h = np.maximum(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter

    return inter / (union + 1e-6)


def diou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise Distance-IoU between two sets of boxes.

    DIoU (Zheng et al., AAAI 2020) subtracts a normalized center
    distance penalty from the IoU:

        DIoU = IoU - d^2 / c^2

    where d is the distance between box centers and c the diagonal of
    the smallest enclosing box. The score lies in (-1, 1]: separated
    boxes keep a monotone ranking gradient (more negative the further
    apart) where plain IoU saturates at 0.

    Args:
        bboxes_a: Bounding boxes, shape (N, 4) in [x1, y1, x2, y2].
        bboxes_b: Bounding boxes, shape (M, 4) in [x1, y1, x2, y2].

    Returns:
        DIoU matrix of shape (N, M).
    """
    if bboxes_a.size == 0 or bboxes_b.size == 0:
        return np.empty((bboxes_a.shape[0], bboxes_b.shape[0]))

    iou = iou_batch(bboxes_a, bboxes_b)

    acx = ((bboxes_a[:, 0] + bboxes_a[:, 2]) / 2.0)[:, np.newaxis]  # (N, 1)
    acy = ((bboxes_a[:, 1] + bboxes_a[:, 3]) / 2.0)[:, np.newaxis]
    bcx = ((bboxes_b[:, 0] + bboxes_b[:, 2]) / 2.0)[np.newaxis, :]  # (1, M)
    bcy = ((bboxes_b[:, 1] + bboxes_b[:, 3]) / 2.0)[np.newaxis, :]
    d2 = (acx - bcx) ** 2 + (acy - bcy) ** 2

    # Smallest enclosing box diagonal
    ew = (
        np.maximum(bboxes_a[:, 2][:, np.newaxis], bboxes_b[:, 2][np.newaxis, :])
        - np.minimum(bboxes_a[:, 0][:, np.newaxis], bboxes_b[:, 0][np.newaxis, :])
    )
    eh = (
        np.maximum(bboxes_a[:, 3][:, np.newaxis], bboxes_b[:, 3][np.newaxis, :])
        - np.minimum(bboxes_a[:, 1][:, np.newaxis], bboxes_b[:, 1][np.newaxis, :])
    )
    c2 = ew ** 2 + eh ** 2

    return iou - d2 / (c2 + 1e-6)


# Sigma-point parameters: canonical unscented transform for n=2, kappa=1
# (gamma = sqrt(n + kappa), w0 = kappa / (n + kappa), wi = 1 / (2(n + kappa)))
SIGMA_GAMMA = float(np.sqrt(3.0))
SIGMA_W0 = 1.0 / 3.0
SIGMA_WI = 1.0 / 6.0


def sigma_diou_batch(
    bboxes_a: np.ndarray,
    bboxes_b: np.ndarray,
    pos_covs: np.ndarray,
    gamma: float = SIGMA_GAMMA,
    max_expansion: float = 1.2,
    pregate: bool = True,
    return_saturation: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Expected DIoU under positional uncertainty (5-point unscented sum).

    Approximates E[DIoU] over each track's predicted-center distribution
    with a 5-point unscented transform: chi_0 is the predicted box and
    chi_{1..4} the box translated by +-gamma * L[:, i], where L is the
    Cholesky factor of the 2x2 positional covariance block (capturing
    Sigma_xy, unlike axis-aligned expansion). The DIoU kernel keeps a
    monotone ranking gradient for widely uncertain tracks where a plain
    IoU sigma sum would saturate at 0.

    The score is accumulated in delta form,

        score = D_0 + sum_i w_i * (D_i - D_0),

    algebraically identical to the weighted sum (the weights sum to 1)
    and bit-exactly equal to ``diou_batch`` when the covariance is zero.

    Per-axis offsets are capped at ``max_expansion`` times the box
    extent, so every translated box stays inside the box expanded by
    ``max_expansion`` per side. Hence pairs with zero ExpansionIoU at
    ``max_expansion`` provably have zero overlap at every sigma point:
    with ``pregate`` enabled those pairs skip the sigma sum and keep the
    kernel's separated value D_0. Pairs that survive the pre-gate score
    identically with or without it.

    Non-PSD covariances (numerical artifacts on long-lost tracks) are
    handled by the closed-form 2x2 Cholesky with clamping: negative
    pivots are treated as zero, degrading gracefully to a diagonal-sqrt
    factor. A genuinely diagonal covariance yields the diagonal-sqrt
    factor exactly, so no separate diagonal fast path is needed.

    Args:
        bboxes_a: Detection boxes, shape (N, 4) in [x1, y1, x2, y2].
        bboxes_b: Predicted track boxes, shape (M, 4).
        pos_covs: Per-track 2x2 positional covariance blocks (cx, cy),
            shape (M, 2, 2).
        gamma: Sigma-point spread. Default sqrt(3) (canonical UT, n=2,
            kappa=1, with w0 = 1/3, wi = 1/6).
        max_expansion: Per-axis offset cap as a fraction of the box
            extent; also the envelope pre-gate expansion.
        pregate: If True, evaluate the sigma sum only for pairs with
            nonzero overlap at maximum expansion.
        return_saturation: If True, also return a boolean mask of shape
            (M,) marking tracks whose plain-IoU kernel is zero at all
            five sigma points against every detection (the
            contrast-collapse case that motivates the DIoU kernel).

    Returns:
        sigma-DIoU matrix of shape (N, M) with values in (-1, 1], or a
        ``(scores, saturated)`` tuple when return_saturation is True.
    """
    num_dets = bboxes_a.shape[0]
    num_trks = bboxes_b.shape[0]
    if bboxes_a.size == 0 or bboxes_b.size == 0:
        scores = np.empty((num_dets, num_trks))
        if return_saturation:
            return scores, np.zeros(num_trks, dtype=bool)
        return scores

    pos_covs = np.asarray(pos_covs, dtype=np.float64)

    # Closed-form 2x2 Cholesky, vectorized over tracks. Clamping guards
    # non-PSD input (negative pivots become zero).
    sxx = np.maximum(pos_covs[:, 0, 0], 0.0)
    syy = np.maximum(pos_covs[:, 1, 1], 0.0)
    sxy = pos_covs[:, 0, 1]
    l11 = np.sqrt(sxx)
    l21 = np.divide(
        sxy, l11, out=np.zeros(num_trks), where=l11 > 0
    )
    l22 = np.sqrt(np.maximum(syy - l21 ** 2, 0.0))

    # Per-axis offset caps keep chi_0 inside the max-expansion envelope
    bw = bboxes_b[:, 2] - bboxes_b[:, 0]
    bh = bboxes_b[:, 3] - bboxes_b[:, 1]
    cap_x = max_expansion * bw
    cap_y = max_expansion * bh
    d1x = np.clip(gamma * l11, -cap_x, cap_x)
    d1y = np.clip(gamma * l21, -cap_y, cap_y)
    d2y = np.clip(gamma * l22, -cap_y, cap_y)

    # Translations chi_{1..4}, shape (4, M): +-column 1, +-column 2
    # (the second Cholesky column has zero x-component)
    zero = np.zeros(num_trks)
    offs_x = np.stack([d1x, -d1x, zero, zero])
    offs_y = np.stack([d1y, -d1y, d2y, -d2y])

    diou0 = diou_batch(bboxes_a, bboxes_b)  # chi_0: kernel at prediction
    scores = diou0.copy()

    if pregate:
        envelope = eiou_batch(
            bboxes_a, bboxes_b, np.full(num_trks, max_expansion)
        )
        pair_mask = envelope > 0.0
        keep = np.flatnonzero(pair_mask.any(axis=0))
    else:
        pair_mask = np.ones((num_dets, num_trks), dtype=bool)
        keep = np.arange(num_trks)

    iou_zero = None
    if return_saturation:
        # Pre-gated pairs provably have zero IoU at every sigma point,
        # so only the evaluated points below can clear a track's flag.
        iou_zero = iou_batch(bboxes_a, bboxes_b) == 0.0

    if keep.size > 0:
        sub_mask = pair_mask[:, keep]
        sub_diou0 = diou0[:, keep]
        for k in range(4):
            shift = np.stack(
                [offs_x[k, keep], offs_y[k, keep],
                 offs_x[k, keep], offs_y[k, keep]],
                axis=-1,
            )
            boxes_k = bboxes_b[keep] + shift
            delta = diou_batch(bboxes_a, boxes_k) - sub_diou0
            scores[:, keep] += SIGMA_WI * np.where(sub_mask, delta, 0.0)
            if iou_zero is not None:
                iou_zero[:, keep] &= iou_batch(bboxes_a, boxes_k) == 0.0

    if return_saturation:
        return scores, iou_zero.all(axis=0)
    return scores


def mahalanobis_distance_batch(
    det_measurements: np.ndarray,
    track_means: np.ndarray,
    track_covariances: np.ndarray,
) -> np.ndarray:
    """Compute pairwise squared Mahalanobis distances in measurement space.

    For each track, the distance uses that track's innovation covariance,
    so the gate adapts to the track's current motion uncertainty (wide for
    maneuvering/occluded tracks, tight for well-observed linear motion).

    A track whose covariance is singular (or numerically non-positive)
    fails open: its distances are returned as 0 so it is never gated out
    by a numerical artifact.

    Args:
        det_measurements: Detection measurements, shape (N, 4) in
            [cx, cy, w, h].
        track_means: Per-track predicted measurement means, shape (M, 4)
            in [cx, cy, w, h].
        track_covariances: Per-track innovation covariances, shape (M, 4, 4).

    Returns:
        Squared Mahalanobis distance matrix of shape (N, M).
    """
    num_dets = det_measurements.shape[0]
    num_trks = track_means.shape[0]
    if num_dets == 0 or num_trks == 0:
        return np.empty((num_dets, num_trks))

    d2 = np.zeros((num_dets, num_trks))
    for j in range(num_trks):
        S = track_covariances[j]
        sign, _ = np.linalg.slogdet(S)
        if sign <= 0:
            logger.debug(f"Non-PSD gating covariance for track column {j}; failing open")
            continue
        S_inv = np.linalg.inv(S)
        diff = det_measurements - track_means[j]  # (N, 4)
        d2[:, j] = np.einsum("ni,ij,nj->n", diff, S_inv, diff)

    return d2


def embedding_similarity_batch(
    det_embs: np.ndarray, trk_embs: np.ndarray
) -> np.ndarray:
    """Compute pairwise cosine similarity between detection and track embeddings.

    Both inputs must be L2-normalized. The dot product of L2-normalized
    vectors equals cosine similarity, producing values in [-1, 1].

    Args:
        det_embs: Detection embeddings, shape (N, D), L2-normalized.
        trk_embs: Track embeddings, shape (M, D), L2-normalized.

    Returns:
        Similarity matrix of shape (N, M). Higher = more similar.
    """
    if det_embs.size == 0 or trk_embs.size == 0:
        return np.empty((det_embs.shape[0], trk_embs.shape[0]))
    return det_embs @ trk_embs.T


def linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    """Solve the linear assignment problem (Hungarian algorithm).

    Finds the optimal one-to-one matching that minimizes total cost.

    Args:
        cost_matrix: Cost matrix of shape (N, M). Lower cost = better match.

    Returns:
        Matched pairs as array of shape (K, 2), where each row is
        [row_index, col_index]. Returns empty (0, 2) array if no
        assignments possible.
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int)

    if _USE_LAP:
        _, row_to_col, _ = lap.lapjv(cost_matrix, extend_cost=True)
        matches = []
        for row_idx, col_idx in enumerate(row_to_col):
            if col_idx >= 0:
                matches.append([row_idx, col_idx])
        if len(matches) == 0:
            return np.empty((0, 2), dtype=int)
        return np.array(matches, dtype=int)
    else:
        row_indices, col_indices = scipy_lsa(cost_matrix)
        return np.column_stack((row_indices, col_indices)).astype(int)
