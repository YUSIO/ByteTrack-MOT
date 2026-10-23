"""Single track state management.

Defines the STrack class which encapsulates all per-track state: Kalman
filter, observation history, lifecycle status, and velocity estimation.
This module depends on kalman_filter.py, imm_kalman_filter.py, and
ocsort_components.py.
"""

from __future__ import annotations

from enum import IntEnum, auto

import numpy as np

from arcsort.kalman_filter import KalmanFilter
from arcsort.ocsort_components import k_previous_obs, speed_direction
from arcsort.utils import xyxy_to_cxcywh, xyxy_to_cxcysr


class TrackState(IntEnum):
    """Lifecycle states for a tracked object."""
    NEW = auto()
    TRACKED = auto()
    LOST = auto()
    REMOVED = auto()


# Global track ID counter
_next_track_id = 0


def reset_track_id() -> None:
    """Reset the global track ID counter. Call between sequences."""
    global _next_track_id
    _next_track_id = 0


class STrack:
    """Single tracked object with Kalman filter state and observation history.

    Manages the full lifecycle of a track from creation through tracking,
    loss, and eventual removal. Stores observation history for OC-SORT's
    observation-centric algorithms (OCM velocity, ORU re-update, OCR recovery).

    Supports both the standard constant-velocity KF and the IMM-KF with
    CV/CA/CT models via the use_imm flag.
    """

    def __init__(
        self,
        bbox: np.ndarray,
        confidence: float,
        delta_t: int = 3,
        use_imm: bool = False,
        imm_config: dict | None = None,
    ) -> None:
        """Initialize a new track from a detection.

        Args:
            bbox: Detection bounding box as [x1, y1, x2, y2], shape (4,).
            confidence: Detection confidence score.
            delta_t: Frames to look back for OCM velocity computation.
            use_imm: If True, use IMM-KF instead of standard KF.
            imm_config: Configuration dict for IMM-KF (ignored if use_imm=False).
        """
        global _next_track_id
        self.track_id = _next_track_id
        _next_track_id += 1

        self.use_imm = use_imm

        if use_imm:
            from arcsort.imm_kalman_filter import IMMKalmanFilter
            self.kf = IMMKalmanFilter(imm_config)
            measurement = xyxy_to_cxcywh(bbox)
        else:
            self.kf = KalmanFilter()
            measurement = xyxy_to_cxcysr(bbox)
        self.kf.initiate(measurement)

        self.state = TrackState.NEW
        self.confidence = confidence
        self.delta_t = delta_t

        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.time_since_update = 0

        # Observation history for OC-SORT algorithms
        self.observations: dict[int, np.ndarray] = {}
        bbox_with_conf = np.concatenate([bbox, [confidence]])
        self.observations[self.age] = bbox_with_conf

        self.last_observation = bbox_with_conf.copy()
        self.velocity = np.array([0.0, 0.0])

        # Appearance embedding for ReID (initialized lazily)
        self.emb: np.ndarray | None = None

    def predict(self) -> np.ndarray:
        """Run Kalman filter prediction step.

        Returns:
            Predicted bounding box as [x1, y1, x2, y2], shape (4,).
        """
        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        return self.get_state()

    def update(
        self,
        bbox: np.ndarray | None,
        confidence: float = 0.0,
        gap_affines: list[np.ndarray] | None = None,
    ) -> None:
        """Update track with a matched detection or mark as unmatched.

        Args:
            bbox: Matched detection as [x1, y1, x2, y2], shape (4,),
                or None if no detection matched this track.
            confidence: Detection confidence score.
            gap_affines: List of 2x3 affine matrices for the observation
                gap period, used by CMC-corrected ORU. None when CMC is
                disabled or track was not lost.
        """
        if bbox is not None:
            # Compute velocity from observation delta_t frames ago
            prev_obs = k_previous_obs(self.observations, self.age, self.delta_t)
            if prev_obs[4] >= 0:
                self.velocity = speed_direction(prev_obs[:4], bbox)
            else:
                self.velocity = np.array([0.0, 0.0])

            bbox_with_conf = np.concatenate([bbox, [confidence]])
            self.observations[self.age] = bbox_with_conf
            self.last_observation = bbox_with_conf.copy()

            if self.use_imm:
                measurement = xyxy_to_cxcywh(bbox)
            else:
                measurement = xyxy_to_cxcysr(bbox)
            self.kf.update(measurement, gap_affines=gap_affines)

            self.hits += 1
            self.hit_streak += 1
            self.time_since_update = 0
            self.confidence = confidence

            if self.state == TrackState.NEW:
                self.state = TrackState.TRACKED
            elif self.state == TrackState.LOST:
                self.state = TrackState.TRACKED
        else:
            self.kf.update(None)

    def mark_lost(self) -> None:
        """Transition track to LOST state."""
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        """Transition track to REMOVED state."""
        self.state = TrackState.REMOVED

    def get_state(self) -> np.ndarray:
        """Get current estimated bounding box.

        Returns:
            Bounding box as [x1, y1, x2, y2], shape (4,).
        """
        return self.kf.get_state_xyxy()

    def get_filter_velocity(self) -> np.ndarray | None:
        """Center velocity [v_cx, v_cy] of the combined filter state.

        Unlike self.velocity (OCM's observation-based unit direction),
        this is the filter's velocity estimate with magnitude — post-CMC
        when CMC is applied before association. None when the underlying
        filter does not expose a combined state (plain KF fallback).
        """
        get_combined = getattr(self.kf, "_get_combined_state", None)
        if get_combined is None:
            return None
        return get_combined()[4:6].copy()

    def apply_cmc(self, affine: np.ndarray) -> None:
        """Apply camera motion compensation to track's Kalman filter state.

        Args:
            affine: 2x3 affine matrix from CMC module.
        """
        self.kf.apply_affine(affine)

    @property
    def is_confirmed(self) -> bool:
        """Whether track has enough hits to be output."""
        return self.state == TrackState.TRACKED

    def update_emb(self, new_emb: np.ndarray, alpha: float = 0.95) -> None:
        """Update appearance embedding via exponential moving average.

        After blending, the embedding is re-normalized to unit length so
        that dot-product similarity remains equivalent to cosine similarity.

        Args:
            new_emb: New detection embedding, shape (D,), L2-normalized.
            alpha: EMA smoothing factor. Higher alpha = more weight on
                the existing (old) embedding. Range [0, 1].
        """
        if self.emb is None:
            self.emb = new_emb.copy()
        else:
            self.emb = alpha * self.emb + (1.0 - alpha) * new_emb
            norm = np.linalg.norm(self.emb)
            if norm > 0:
                self.emb /= norm

    def get_emb(self) -> np.ndarray | None:
        """Get current appearance embedding.

        Returns:
            L2-normalized embedding of shape (D,), or None if not set.
        """
        return self.emb

    def get_previous_obs(self) -> np.ndarray:
        """Get observation from delta_t frames ago for OCM.

        Returns:
            Previous observation as [x1, y1, x2, y2, conf], shape (5,).
        """
        return k_previous_obs(self.observations, self.age, self.delta_t)
