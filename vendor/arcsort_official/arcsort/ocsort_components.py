"""OC-SORT specific algorithms: ORU, OCM, and OCR.

Implements the three observation-centric innovations from OC-SORT:
    - OCM (Observation-Centric Momentum): direction consistency scoring
      using actual past observations instead of filter predictions.
    - ORU (Observation-centric Re-Update): virtual trajectory generation
      for re-updating the Kalman filter after an observation gap. The
      actual re-update logic lives in KalmanFilter._unfreeze().
    - OCR (Observation-Centric Recovery): second-chance matching using
      last observed position instead of predicted position.

Reference: Cao et al., "Observation-Centric SORT", CVPR 2023.
"""

from __future__ import annotations

import numpy as np


def speed_direction(bbox_prev: np.ndarray, bbox_curr: np.ndarray) -> np.ndarray:
    """Compute normalized direction vector between two bounding box centers.

    Args:
        bbox_prev: Previous bbox as [x1, y1, x2, y2], shape (4,).
        bbox_curr: Current bbox as [x1, y1, x2, y2], shape (4,).

    Returns:
        Normalized direction [dy, dx], shape (2,). Returns [0, 0] if
        the two centers coincide.
    """
    cx_prev = (bbox_prev[0] + bbox_prev[2]) / 2
    cy_prev = (bbox_prev[1] + bbox_prev[3]) / 2
    cx_curr = (bbox_curr[0] + bbox_curr[2]) / 2
    cy_curr = (bbox_curr[1] + bbox_curr[3]) / 2

    dy = cy_curr - cy_prev
    dx = cx_curr - cx_prev
    norm = np.sqrt(dy * dy + dx * dx) + 1e-6

    return np.array([dy / norm, dx / norm])


def speed_direction_batch(
    detections: np.ndarray,
    previous_obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute direction vectors from previous observations to detections.

    Vectorized version for computing direction consistency in association.

    Args:
        detections: Detection bboxes, shape (N, 4) in [x1, y1, x2, y2].
        previous_obs: Previous observation bboxes for tracks, shape (M, 4)
            in [x1, y1, x2, y2].

    Returns:
        Tuple of (dy, dx) arrays, each shape (M, N), representing the
        normalized direction from each track's previous observation
        to each detection.
    """
    det_cx = (detections[:, 0] + detections[:, 2]) / 2
    det_cy = (detections[:, 1] + detections[:, 3]) / 2
    obs_cx = (previous_obs[:, 0] + previous_obs[:, 2]) / 2
    obs_cy = (previous_obs[:, 1] + previous_obs[:, 3]) / 2

    # Broadcast: (M, 1) - (1, N) -> (M, N)
    dy = det_cy[np.newaxis, :] - obs_cy[:, np.newaxis]
    dx = det_cx[np.newaxis, :] - obs_cx[:, np.newaxis]
    norm = np.sqrt(dy * dy + dx * dx) + 1e-6

    return dy / norm, dx / norm


def k_previous_obs(
    observations: dict[int, np.ndarray],
    current_age: int,
    delta_t: int,
) -> np.ndarray:
    """Retrieve observation from delta_t frames ago.

    Used by OCM to compute velocity direction from observation history
    rather than from Kalman filter state.

    Args:
        observations: Dict mapping frame age to observed bbox [x1, y1, x2, y2, conf].
        current_age: Current age of the track (frame count).
        delta_t: Number of frames to look back.

    Returns:
        Observation bbox as [x1, y1, x2, y2, conf], shape (5,).
        Returns [-1, -1, -1, -1, -1] if no valid observation found.
    """
    # Try exact delta_t frames ago
    target_age = current_age - delta_t
    if target_age in observations:
        return observations[target_age]

    # Fall back to most recent observation before target_age
    valid_ages = [age for age in observations if age <= target_age]
    if valid_ages:
        return observations[max(valid_ages)]

    # No valid previous observation
    return np.array([-1.0, -1.0, -1.0, -1.0, -1.0])


def compute_ocm_cost(
    detections: np.ndarray,
    track_velocities: np.ndarray,
    track_previous_obs: np.ndarray,
    inertia_weight: float,
) -> np.ndarray:
    """Compute OCM direction consistency cost matrix.

    Measures how well the direction from a track's previous observation
    to a candidate detection aligns with the track's recent velocity
    direction. High alignment (low angle difference) indicates consistent
    motion, producing lower cost.

    Args:
        detections: Detection bboxes, shape (N, 4) in [x1, y1, x2, y2].
        track_velocities: Per-track velocity directions, shape (M, 2)
            as [dy, dx] (normalized).
        track_previous_obs: Per-track previous observation bboxes,
            shape (M, 5) in [x1, y1, x2, y2, conf].
        inertia_weight: Weight for the direction consistency cost.

    Returns:
        Cost matrix of shape (N, M). Higher values penalize direction
        inconsistency. Zero for tracks without valid previous observations.
    """
    num_dets = detections.shape[0]
    num_trks = track_velocities.shape[0]

    if num_dets == 0 or num_trks == 0:
        return np.zeros((num_dets, num_trks))

    # Direction from each track's previous obs to each detection
    dy, dx = speed_direction_batch(detections, track_previous_obs[:, :4])

    # Track inertia directions
    inertia_dy = track_velocities[:, 0]
    inertia_dx = track_velocities[:, 1]

    # Cosine of angle between inertia and candidate direction
    # dot product: (M, N)
    cos_angle = inertia_dy[:, np.newaxis] * dy + inertia_dx[:, np.newaxis] * dx
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Convert to angle difference in [0, 1] range
    # 0 = perfectly aligned, 1 = opposite direction
    angle_diff = np.arccos(cos_angle)
    angle_diff = (np.pi / 2 - np.abs(angle_diff)) / np.pi

    # Mask out tracks with no valid previous observation (conf == -1)
    valid_mask = (track_previous_obs[:, 4] >= 0).astype(np.float64)
    angle_diff = angle_diff * valid_mask[:, np.newaxis]

    # Transpose to (N_det, M_trk) and apply weight
    return (angle_diff * inertia_weight).T
