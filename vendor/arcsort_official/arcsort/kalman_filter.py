"""Standard Kalman filter for single-object state estimation.

Implements a constant-velocity Kalman filter operating in [cx, cy, s, r]
space (center, scale/area, aspect ratio). This is the baseline filter
from OC-SORT; the IMM-KF (imm_kalman_filter.py) replaces it when use_imm is set.

State vector (7D): [cx, cy, s, r, v_cx, v_cy, v_s]
Measurement vector (4D): [cx, cy, s, r]

The aspect ratio velocity is not modeled (assumed constant) to reduce
state dimension and improve stability for pedestrian/person tracking.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np


STATE_DIM = 7
MEASUREMENT_DIM = 4


class KalmanFilter:
    """Constant-velocity Kalman filter in [cx, cy, s, r] space.

    Supports freeze/unfreeze mechanism for OC-SORT's observation-centric
    re-update (ORU). When a track loses observations, the filter state is
    frozen. Upon re-observation, virtual observations are generated via
    linear interpolation and the filter is re-updated.
    """

    def __init__(self) -> None:
        self.x = np.zeros((STATE_DIM, 1))
        self.P = np.eye(STATE_DIM)

        # State transition: constant velocity model
        self.F = np.eye(STATE_DIM)
        self.F[0, 4] = 1.0  # cx += v_cx
        self.F[1, 5] = 1.0  # cy += v_cy
        self.F[2, 6] = 1.0  # s  += v_s

        # Measurement: observe [cx, cy, s, r] directly
        self.H = np.zeros((MEASUREMENT_DIM, STATE_DIM))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # Measurement noise
        self.R = np.eye(MEASUREMENT_DIM)
        self.R[2:, 2:] *= 10.0

        # Process noise
        self.Q = np.eye(STATE_DIM)
        self.Q[4:, 4:] *= 0.01
        self.Q[-1, -1] *= 0.01

        # Initial state covariance
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0

        self._frozen_state: dict | None = None
        self.history_obs: list[np.ndarray | None] = []

    def initiate(self, measurement: np.ndarray) -> None:
        """Initialize filter state from first measurement.

        Args:
            measurement: First observation as [cx, cy, s, r], shape (4,).
        """
        self.x[:MEASUREMENT_DIM, 0] = measurement
        # Velocities remain zero (no initial velocity estimate)

    def predict(self) -> np.ndarray:
        """Run prediction step.

        Clamps scale velocity to prevent negative area predictions.

        Returns:
            Predicted state as [cx, cy, s, r, v_cx, v_cy, v_s], shape (7,).
        """
        # Prevent negative scale: if predicted area would be negative, zero velocity
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] *= 0.0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x[:, 0].copy()

    def update(
        self,
        measurement: np.ndarray | None,
        gap_affines: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        """Run measurement update step.

        Handles missing observations via freeze/unfreeze for ORU support.

        Args:
            measurement: Observation as [cx, cy, s, r], shape (4,), or None
                if no detection matched this track.
            gap_affines: List of 2x3 affine matrices for frames during the
                observation gap, in chronological order. Used by CMC-corrected
                ORU to warp virtual observations into the correct coordinate
                frame. None to use original (uncorrected) ORU.

        Returns:
            Updated state as [cx, cy, s, r, v_cx, v_cy, v_s], shape (7,).
        """
        self.history_obs.append(measurement)

        if measurement is None:
            if self._frozen_state is None:
                self._freeze()
            return self.x[:, 0].copy()

        if self._frozen_state is not None:
            self._unfreeze(measurement, gap_affines)

        self._kalman_update(measurement)
        return self.x[:, 0].copy()

    def _kalman_update(self, measurement: np.ndarray) -> None:
        """Apply standard Kalman measurement update equations.

        Uses the Joseph form for numerical stability:
            P = (I - KH) P (I - KH)' + K R K'

        Args:
            measurement: Observation as [cx, cy, s, r], shape (4,).
        """
        z = measurement.reshape(MEASUREMENT_DIM, 1)

        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form)
        I_KH = np.eye(STATE_DIM) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    def _freeze(self) -> None:
        """Save filter state before entering observation gap.

        Called when the first None measurement arrives after observed frames.
        Stores everything needed to reconstruct state for ORU re-update.
        """
        self._frozen_state = {
            "x": self.x.copy(),
            "P": self.P.copy(),
            "history_obs": deepcopy(self.history_obs),
        }

    @staticmethod
    def _apply_affine_to_obs_cxcysr(
        obs: np.ndarray, affine: np.ndarray
    ) -> np.ndarray:
        """Apply 2x3 affine transform to an observation in [cx,cy,s,r] space.

        Args:
            obs: Observation as [cx, cy, s, r], shape (4,).
            affine: 2x3 affine matrix.

        Returns:
            Transformed observation, shape (4,).
        """
        R = affine[:2, :2]
        t = affine[:2, 2]
        det_R = np.linalg.det(R)
        cx, cy, s, r = obs
        new_center = R @ np.array([cx, cy]) + t
        return np.array([new_center[0], new_center[1], s * det_R, r])

    def _unfreeze(
        self,
        new_measurement: np.ndarray,
        gap_affines: list[np.ndarray] | None = None,
    ) -> None:
        """Restore frozen state and apply ORU virtual trajectory re-update.

        Generates linearly interpolated virtual observations between the
        last real observation before the gap and the new observation after
        the gap, then re-runs predict/update for each virtual frame.

        When gap_affines are provided (CMC-corrected ORU), the interpolation
        is done in the frozen frame's coordinate system, then each virtual
        observation is transformed to its target frame using the cumulative
        affine. The KF state is also CMC-warped at each step.

        Args:
            new_measurement: The new observation that ended the gap,
                as [cx, cy, s, r], shape (4,).
            gap_affines: List of 2x3 affine matrices, one per gap frame,
                in chronological order. None for uncorrected ORU.
        """
        frozen = self._frozen_state
        if frozen is None:
            return

        # Restore state to the point just before observations were lost
        self.x = frozen["x"]
        self.P = frozen["P"]
        old_history = frozen["history_obs"]

        # Find the last real observation before the gap
        last_obs = None
        last_obs_idx = -1
        for idx in range(len(old_history) - 1, -1, -1):
            if old_history[idx] is not None:
                last_obs = old_history[idx]
                last_obs_idx = idx
                break

        if last_obs is None:
            self._frozen_state = None
            return

        # Count missing frames (None entries after last real observation)
        gap_length = len(self.history_obs) - 1 - last_obs_idx

        if gap_length > 0:
            use_cmc = (
                gap_affines is not None and len(gap_affines) >= gap_length
            )

            if use_cmc:
                # CMC-corrected ORU: interpolate in frozen frame, transform
                # each virtual observation to its target frame
                # Compute full cumulative affine (frozen frame -> current)
                cumul_3x3 = np.eye(3)
                for aff in gap_affines[:gap_length]:
                    aff_3x3 = np.eye(3)
                    aff_3x3[:2] = aff
                    cumul_3x3 = aff_3x3 @ cumul_3x3
                inv_cumul = np.linalg.inv(cumul_3x3)

                # Transform new_measurement back to frozen frame for interp
                new_obs_frozen = self._apply_affine_to_obs_cxcysr(
                    new_measurement, inv_cumul[:2]
                )

                # Replay with per-step CMC correction
                partial_3x3 = np.eye(3)
                for step in range(1, gap_length + 1):
                    alpha = step / gap_length
                    # Interpolate in frozen frame
                    virtual_frozen = (
                        last_obs * (1.0 - alpha) + new_obs_frozen * alpha
                    )

                    # Accumulate affine to this step's frame
                    aff = gap_affines[step - 1]
                    aff_3x3 = np.eye(3)
                    aff_3x3[:2] = aff
                    partial_3x3 = aff_3x3 @ partial_3x3

                    # Transform virtual obs to this step's frame
                    virtual_obs = self._apply_affine_to_obs_cxcysr(
                        virtual_frozen, partial_3x3[:2]
                    )

                    self.predict()
                    self.apply_affine(aff)
                    self._kalman_update(virtual_obs)
            else:
                # Original ORU: interpolate in pixel space (no CMC)
                for step in range(1, gap_length + 1):
                    alpha = step / gap_length
                    virtual_obs = (
                        last_obs * (1.0 - alpha) + new_measurement * alpha
                    )
                    self.predict()
                    self._kalman_update(virtual_obs)

        self._frozen_state = None

    def apply_affine(self, affine: np.ndarray) -> None:
        """Apply a 2D affine transform to the filter state and covariance.

        Used by Camera Motion Compensation (CMC) to warp predictions into
        the current frame's coordinate system after camera movement.

        For our state [cx, cy, s, r, v_cx, v_cy, v_s]:
        - (cx, cy): position, transformed by full affine (R @ pos + t)
        - s (area): scaled by det(R) (area scaling factor)
        - r (aspect ratio): invariant under similarity transforms
        - (v_cx, v_cy): velocity, rotated by R (no translation)
        - v_s: area velocity, scaled by det(R)

        Args:
            affine: 2x3 affine matrix from CMC, shape (2, 3).
        """
        R = affine[:2, :2]
        t = affine[:2, 2]
        det_R = np.linalg.det(R)

        # Build 7x7 state transform matrix
        T = np.zeros((STATE_DIM, STATE_DIM))
        # Position block: R @ [cx, cy]
        T[0:2, 0:2] = R
        # Area: s * det(R)
        T[2, 2] = det_R
        # Aspect ratio: unchanged
        T[3, 3] = 1.0
        # Velocity block: R @ [v_cx, v_cy]
        T[4:6, 4:6] = R
        # Area velocity: v_s * det(R)
        T[6, 6] = det_R

        # Translation vector (only position gets translated)
        b = np.zeros((STATE_DIM, 1))
        b[0, 0] = t[0]
        b[1, 0] = t[1]

        # Transform state and covariance
        self.x = T @ self.x + b
        self.P = T @ self.P @ T.T

    def get_velocity(self) -> np.ndarray:
        """Get the estimated center velocity.

        Returns:
            Velocity [v_cx, v_cy] in pixels per frame, shape (2,).
        """
        return self.x[4:6, 0].copy()

    def get_state_xyxy(self) -> np.ndarray:
        """Get current state estimate as [x1, y1, x2, y2] bbox.

        Returns:
            Bounding box in xyxy format, shape (4,).
        """
        cx, cy, s, r = self.x[:4, 0]
        w = np.sqrt(max(s * r, 0))
        h = s / (w + 1e-6)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
