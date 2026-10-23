"""Interacting Multiple Model Kalman Filter (IMM-KF).

Runs three motion models in parallel -- Constant Velocity (CV), Constant
Acceleration (CA), and Coordinated Turn (CT) -- and fuses their predictions
based on measurement likelihoods. This enables adaptive tracking of objects
with non-linear motion (athletes, dancers, gymnasts).

All models operate in [cx, cy, w, h] space. The CT model uses an Extended
Kalman Filter (EKF) for its non-linear state transition.

Reference: Bar-Shalom, Li, Kirubarajan, "Estimation with Applications to
Tracking and Navigation", Ch. 11.6 (IMM algorithm).
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from loguru import logger

from arcsort.utils import cxcywh_to_xyxy


# State dimensions per model
CV_STATE_DIM = 8   # [cx, cy, w, h, v_cx, v_cy, v_w, v_h]
CA_STATE_DIM = 10  # [cx, cy, w, h, v_cx, v_cy, v_w, v_h, a_cx, a_cy]
CT_STATE_DIM = 9   # [cx, cy, w, h, v_cx, v_cy, v_w, v_h, omega]
MEASUREMENT_DIM = 4  # [cx, cy, w, h]
COMMON_DIM = 8     # Shared across all models: [cx, cy, w, h, v_cx, v_cy, v_w, v_h]

# Small value to prevent division by zero in CT model
OMEGA_EPS = 1e-6

# Size-dependent noise weights (BoT-SORT pattern)
# Noise std = weight * object_size for each state dim
STD_WEIGHT_POSITION = 1.0 / 20   # 0.05
STD_WEIGHT_VELOCITY = 1.0 / 160  # 0.00625

# Default IMM configuration
DEFAULT_IMM_CONFIG = {
    "models": ["cv", "ca", "ct"],
    "transition_prob_diagonal": 0.90,
    "process_noise_cv": 1.0,
    "process_noise_ca": 1.0,
    "process_noise_ct": 1.0,
    "initial_model_probs": [0.5, 0.25, 0.25],
}


def _log_likelihood(innovation: np.ndarray, S: np.ndarray) -> float:
    """Compute log-likelihood of innovation under covariance S.

    Args:
        innovation: Innovation vector, shape (n, 1).
        S: Innovation covariance, shape (n, n).

    Returns:
        Log-likelihood value.
    """
    n = innovation.shape[0]
    sign, log_det = np.linalg.slogdet(S)
    if sign <= 0:
        return -1e10
    mahal = float((innovation.T @ np.linalg.inv(S) @ innovation).item())
    return -0.5 * (n * np.log(2 * np.pi) + log_det + mahal)


def _build_projection_matrix(model_dim: int) -> np.ndarray:
    """Build matrix to project model state to common 8-dim subspace.

    The common subspace is the first 8 dimensions of every model,
    which are [cx, cy, w, h, v_cx, v_cy, v_w, v_h].

    Args:
        model_dim: Full state dimension of the model.

    Returns:
        Projection matrix of shape (COMMON_DIM, model_dim).
    """
    G = np.zeros((COMMON_DIM, model_dim))
    G[:COMMON_DIM, :COMMON_DIM] = np.eye(COMMON_DIM)
    return G


def _build_injection_matrix(model_dim: int) -> np.ndarray:
    """Build matrix to inject common 8-dim state back into model state.

    Args:
        model_dim: Full state dimension of the model.

    Returns:
        Injection matrix of shape (model_dim, COMMON_DIM).
    """
    G_plus = np.zeros((model_dim, COMMON_DIM))
    G_plus[:COMMON_DIM, :COMMON_DIM] = np.eye(COMMON_DIM)
    return G_plus


class CVModel:
    """Constant Velocity motion model (8-dim state).

    State: [cx, cy, w, h, v_cx, v_cy, v_w, v_h]
    Assumes all state variables change at constant rate.
    """

    def __init__(self, process_noise_scale: float = 1.0, dt: float = 1.0) -> None:
        self.dim = CV_STATE_DIM
        self.dt = dt
        self.process_noise_scale = process_noise_scale

        self.x = np.zeros((self.dim, 1))
        self.P = np.eye(self.dim)

        # Transition: position += velocity * dt
        self.F = np.eye(self.dim)
        for i in range(4):
            self.F[i, i + 4] = dt

        # Observation: measure [cx, cy, w, h] directly
        self.H = np.zeros((MEASUREMENT_DIM, self.dim))
        self.H[:4, :4] = np.eye(4)

        self.Q = np.eye(self.dim)
        self.R = np.eye(MEASUREMENT_DIM)

    def _compute_process_noise(self) -> np.ndarray:
        """Compute size-dependent process noise (BoT-SORT pattern)."""
        w = max(float(self.x[2, 0]), 1.0)
        h = max(float(self.x[3, 0]), 1.0)
        s = self.process_noise_scale
        std_pos = s * np.array([
            STD_WEIGHT_POSITION * w,
            STD_WEIGHT_POSITION * h,
            STD_WEIGHT_POSITION * w,
            STD_WEIGHT_POSITION * h,
        ])
        std_vel = s * np.array([
            STD_WEIGHT_VELOCITY * w,
            STD_WEIGHT_VELOCITY * h,
            STD_WEIGHT_VELOCITY * w,
            STD_WEIGHT_VELOCITY * h,
        ])
        return np.diag(np.concatenate([std_pos, std_vel]) ** 2)

    def _compute_measurement_noise(self) -> np.ndarray:
        """Compute size-dependent measurement noise."""
        w = max(float(self.x[2, 0]), 1.0)
        h = max(float(self.x[3, 0]), 1.0)
        std = np.array([
            STD_WEIGHT_POSITION * w,
            STD_WEIGHT_POSITION * h,
            STD_WEIGHT_POSITION * w,
            STD_WEIGHT_POSITION * h,
        ])
        return np.diag(std ** 2)

    def predict(self) -> None:
        """Linear predict step with size-dependent process noise."""
        self.Q = self._compute_process_noise()
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> float:
        """Linear update step.

        Args:
            z: Measurement [cx, cy, w, h], shape (4, 1).

        Returns:
            Log-likelihood of the measurement given the predicted state.
        """
        self.R = self._compute_measurement_noise()

        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        log_lik = _log_likelihood(innovation, S)

        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        I_KH = np.eye(self.dim) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return log_lik


class CAModel:
    """Constant Acceleration motion model (10-dim state).

    State: [cx, cy, w, h, v_cx, v_cy, v_w, v_h, a_cx, a_cy]
    Center position has acceleration terms; width/height remain CV.
    """

    def __init__(self, process_noise_scale: float = 1.0, dt: float = 1.0) -> None:
        self.dim = CA_STATE_DIM
        self.dt = dt
        self.process_noise_scale = process_noise_scale

        self.x = np.zeros((self.dim, 1))
        self.P = np.eye(self.dim)

        # Transition matrix
        self.F = np.eye(self.dim)
        # Position += velocity * dt
        for i in range(4):
            self.F[i, i + 4] = dt
        # cx += 0.5 * a_cx * dt^2, cy += 0.5 * a_cy * dt^2
        self.F[0, 8] = 0.5 * dt ** 2
        self.F[1, 9] = 0.5 * dt ** 2
        # v_cx += a_cx * dt, v_cy += a_cy * dt
        self.F[4, 8] = dt
        self.F[5, 9] = dt

        # Observation
        self.H = np.zeros((MEASUREMENT_DIM, self.dim))
        self.H[:4, :4] = np.eye(4)

        self.Q = np.eye(self.dim)
        self.R = np.eye(MEASUREMENT_DIM)

    def _compute_process_noise(self) -> np.ndarray:
        w = max(float(self.x[2, 0]), 1.0)
        h = max(float(self.x[3, 0]), 1.0)
        s = self.process_noise_scale
        std_pos = s * np.array([
            STD_WEIGHT_POSITION * w, STD_WEIGHT_POSITION * h,
            STD_WEIGHT_POSITION * w, STD_WEIGHT_POSITION * h,
        ])
        std_vel = s * np.array([
            STD_WEIGHT_VELOCITY * w, STD_WEIGHT_VELOCITY * h,
            STD_WEIGHT_VELOCITY * w, STD_WEIGHT_VELOCITY * h,
        ])
        # Acceleration noise: proportional to size, smaller than velocity
        std_accel = s * np.array([
            STD_WEIGHT_VELOCITY * w * 0.5,
            STD_WEIGHT_VELOCITY * h * 0.5,
        ])
        return np.diag(np.concatenate([std_pos, std_vel, std_accel]) ** 2)

    def _compute_measurement_noise(self) -> np.ndarray:
        w = max(float(self.x[2, 0]), 1.0)
        h = max(float(self.x[3, 0]), 1.0)
        std = np.array([
            STD_WEIGHT_POSITION * w, STD_WEIGHT_POSITION * h,
            STD_WEIGHT_POSITION * w, STD_WEIGHT_POSITION * h,
        ])
        return np.diag(std ** 2)

    def predict(self) -> None:
        self.Q = self._compute_process_noise()
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> float:
        self.R = self._compute_measurement_noise()

        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        log_lik = _log_likelihood(innovation, S)

        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        I_KH = np.eye(self.dim) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return log_lik


class CTModel:
    """Coordinated Turn motion model (9-dim state, EKF).

    State: [cx, cy, w, h, v_cx, v_cy, v_w, v_h, omega]
    Center position follows a curved path parameterized by turn rate omega.
    Width/height remain constant-velocity.
    """

    def __init__(self, process_noise_scale: float = 1.0, dt: float = 1.0) -> None:
        self.dim = CT_STATE_DIM
        self.dt = dt
        self.process_noise_scale = process_noise_scale

        self.x = np.zeros((self.dim, 1))
        self.P = np.eye(self.dim)

        # Observation is linear
        self.H = np.zeros((MEASUREMENT_DIM, self.dim))
        self.H[:4, :4] = np.eye(4)

        self.Q = np.eye(self.dim)
        self.R = np.eye(MEASUREMENT_DIM)

    def _compute_process_noise(self) -> np.ndarray:
        w = max(float(self.x[2, 0]), 1.0)
        h = max(float(self.x[3, 0]), 1.0)
        s = self.process_noise_scale
        std_pos = s * np.array([
            STD_WEIGHT_POSITION * w, STD_WEIGHT_POSITION * h,
            STD_WEIGHT_POSITION * w, STD_WEIGHT_POSITION * h,
        ])
        std_vel = s * np.array([
            STD_WEIGHT_VELOCITY * w, STD_WEIGHT_VELOCITY * h,
            STD_WEIGHT_VELOCITY * w, STD_WEIGHT_VELOCITY * h,
        ])
        # Omega noise: small, turn rate changes slowly
        std_omega = np.array([s * 0.001])
        return np.diag(np.concatenate([std_pos, std_vel, std_omega]) ** 2)

    def _compute_measurement_noise(self) -> np.ndarray:
        w = max(float(self.x[2, 0]), 1.0)
        h = max(float(self.x[3, 0]), 1.0)
        std = np.array([
            STD_WEIGHT_POSITION * w, STD_WEIGHT_POSITION * h,
            STD_WEIGHT_POSITION * w, STD_WEIGHT_POSITION * h,
        ])
        return np.diag(std ** 2)

    def _transition(self, x: np.ndarray) -> np.ndarray:
        """Non-linear state transition for coordinated turn.

        Args:
            x: Current state, shape (9, 1).

        Returns:
            Predicted state, shape (9, 1).
        """
        dt = self.dt
        cx, cy, w, h = x[0, 0], x[1, 0], x[2, 0], x[3, 0]
        vx, vy = x[4, 0], x[5, 0]
        vw, vh = x[6, 0], x[7, 0]
        omega = x[8, 0]

        x_pred = np.zeros_like(x)

        if abs(omega) < OMEGA_EPS:
            # Near-zero turn rate: use Taylor expansion (reduces to CV)
            x_pred[0, 0] = cx + vx * dt
            x_pred[1, 0] = cy + vy * dt
            x_pred[4, 0] = vx
            x_pred[5, 0] = vy
        else:
            sin_wdt = np.sin(omega * dt)
            cos_wdt = np.cos(omega * dt)
            x_pred[0, 0] = cx + sin_wdt / omega * vx - (1 - cos_wdt) / omega * vy
            x_pred[1, 0] = cy + (1 - cos_wdt) / omega * vx + sin_wdt / omega * vy
            x_pred[4, 0] = cos_wdt * vx - sin_wdt * vy
            x_pred[5, 0] = sin_wdt * vx + cos_wdt * vy

        # w, h evolve as constant velocity (linear)
        x_pred[2, 0] = w + vw * dt
        x_pred[3, 0] = h + vh * dt
        x_pred[6, 0] = vw
        x_pred[7, 0] = vh
        # omega is constant
        x_pred[8, 0] = omega

        return x_pred

    def _jacobian(self, x: np.ndarray) -> np.ndarray:
        """Compute Jacobian of the transition function.

        Args:
            x: Current state, shape (9, 1).

        Returns:
            Jacobian matrix, shape (9, 9).
        """
        dt = self.dt
        vx, vy = x[4, 0], x[5, 0]
        omega = x[8, 0]

        F = np.eye(self.dim)
        # w, h block: linear CV
        F[2, 6] = dt  # w += vw * dt
        F[3, 7] = dt  # h += vh * dt

        if abs(omega) < OMEGA_EPS:
            # Taylor expansion at omega=0
            F[0, 4] = dt       # d(cx)/d(vx)
            F[1, 5] = dt       # d(cy)/d(vy)
            # d(cx)/d(omega) and d(cy)/d(omega) via Taylor:
            # cx = cx + vx*dt - 0.5*vy*omega*dt^2 + ...
            F[0, 8] = -0.5 * vy * dt ** 2
            F[1, 8] = 0.5 * vx * dt ** 2
        else:
            sin_wdt = np.sin(omega * dt)
            cos_wdt = np.cos(omega * dt)
            w = omega

            # d(cx)/d(vx), d(cx)/d(vy)
            F[0, 4] = sin_wdt / w
            F[0, 5] = -(1 - cos_wdt) / w
            # d(cy)/d(vx), d(cy)/d(vy)
            F[1, 4] = (1 - cos_wdt) / w
            F[1, 5] = sin_wdt / w

            # d(vx)/d(vx), d(vx)/d(vy)
            F[4, 4] = cos_wdt
            F[4, 5] = -sin_wdt
            # d(vy)/d(vx), d(vy)/d(vy)
            F[5, 4] = sin_wdt
            F[5, 5] = cos_wdt

            # d(cx)/d(omega)
            F[0, 8] = (
                (cos_wdt * dt * w - sin_wdt) / w ** 2 * vx
                - (sin_wdt * dt * w - (1 - cos_wdt)) / w ** 2 * vy
            )
            # d(cy)/d(omega)
            F[1, 8] = (
                (sin_wdt * dt * w - (1 - cos_wdt)) / w ** 2 * vx
                + (cos_wdt * dt * w - sin_wdt) / w ** 2 * vy
            )
            # d(vx)/d(omega)
            F[4, 8] = -sin_wdt * dt * vx - cos_wdt * dt * vy
            # d(vy)/d(omega)
            F[5, 8] = cos_wdt * dt * vx - sin_wdt * dt * vy

        return F

    def predict(self) -> None:
        """EKF predict: non-linear transition + Jacobian for covariance."""
        self.Q = self._compute_process_noise()
        F_jac = self._jacobian(self.x)
        self.x = self._transition(self.x)
        self.P = F_jac @ self.P @ F_jac.T + self.Q

    def update(self, z: np.ndarray) -> float:
        """Linear update (observation model is linear)."""
        self.R = self._compute_measurement_noise()

        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        log_lik = _log_likelihood(innovation, S)

        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        I_KH = np.eye(self.dim) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return log_lik


# Model registry for factory construction
_MODEL_CLASSES = {
    "cv": (CVModel, CV_STATE_DIM),
    "ca": (CAModel, CA_STATE_DIM),
    "ct": (CTModel, CT_STATE_DIM),
}


class IMMKalmanFilter:
    """Interacting Multiple Model Kalman Filter.

    Runs multiple motion models in parallel and fuses their predictions
    based on measurement likelihoods. Provides the same public interface
    as the standard KalmanFilter for drop-in replacement.

    The IMM algorithm has three steps per cycle:
    1. Interaction (mixing): blend model states based on transition probs
    2. Model-conditional filtering: each model predicts/updates independently
    3. Combination: fuse results weighted by updated model probabilities
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize IMM-KF with configuration.

        Args:
            config: IMM configuration dictionary. Uses DEFAULT_IMM_CONFIG
                for any missing keys.
        """
        if config is None:
            config = {}
        cfg = {**DEFAULT_IMM_CONFIG, **config}

        model_names = cfg["models"]
        num_models = len(model_names)

        # Build models
        noise_keys = {
            "cv": "process_noise_cv",
            "ca": "process_noise_ca",
            "ct": "process_noise_ct",
        }
        self.models: list[CVModel | CAModel | CTModel] = []
        for name in model_names:
            cls, _ = _MODEL_CLASSES[name]
            noise_scale = cfg[noise_keys[name]]
            self.models.append(cls(process_noise_scale=noise_scale))

        self.num_models = num_models
        self.model_names = model_names

        # Model probabilities
        probs = np.array(cfg["initial_model_probs"], dtype=np.float64)
        self.model_probs = probs / probs.sum()
        self._initial_model_probs = self.model_probs.copy()

        # Transition probability matrix (Markov chain)
        diag = cfg["transition_prob_diagonal"]
        if num_models > 1:
            off_diag = (1.0 - diag) / (num_models - 1)
        else:
            off_diag = 0.0
        self.transition_prob = np.full((num_models, num_models), off_diag)
        np.fill_diagonal(self.transition_prob, diag)

        # Projection/injection matrices for dimension mapping
        self._G = [_build_projection_matrix(m.dim) for m in self.models]
        self._G_plus = [_build_injection_matrix(m.dim) for m in self.models]

        # ORU support
        self._frozen_state: dict | None = None
        self.history_obs: list[np.ndarray | None] = []

    def initiate(self, measurement: np.ndarray) -> None:
        """Initialize all models from first measurement.

        Uses BoT-SORT's size-dependent initial covariance pattern.

        Args:
            measurement: First observation as [cx, cy, w, h], shape (4,).
        """
        w_val = max(float(measurement[2]), 1.0)
        h_val = max(float(measurement[3]), 1.0)

        for model in self.models:
            model.x[:MEASUREMENT_DIM, 0] = measurement
            # Velocities and model-specific dims stay zero

            # Size-dependent initial covariance (BoT-SORT pattern)
            std_pos = [
                2 * STD_WEIGHT_POSITION * w_val,
                2 * STD_WEIGHT_POSITION * h_val,
                2 * STD_WEIGHT_POSITION * w_val,
                2 * STD_WEIGHT_POSITION * h_val,
            ]
            std_vel = [
                10 * STD_WEIGHT_VELOCITY * w_val,
                10 * STD_WEIGHT_VELOCITY * h_val,
                10 * STD_WEIGHT_VELOCITY * w_val,
                10 * STD_WEIGHT_VELOCITY * h_val,
            ]
            std_all = std_pos + std_vel
            # Model-specific dims: large initial uncertainty
            for _ in range(model.dim - COMMON_DIM):
                std_all.append(1.0)
            model.P = np.diag(np.array(std_all) ** 2)

    def predict(self) -> np.ndarray:
        """Run IMM interaction step then per-model prediction.

        Returns:
            Combined predicted state (unused by caller, but matches
            KalmanFilter interface). Shape varies.
        """
        self._imm_interact()

        for model in self.models:
            model.predict()
            # Clamp width/height to prevent negative values
            model.x[2, 0] = max(model.x[2, 0], 1.0)
            model.x[3, 0] = max(model.x[3, 0], 1.0)

        return self._get_combined_state()

    def update(
        self,
        measurement: np.ndarray | None,
        gap_affines: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        """Run per-model update then IMM combination.

        Args:
            measurement: Observation as [cx, cy, w, h], shape (4,),
                or None if no detection matched this track.
            gap_affines: List of 2x3 affine matrices for frames during the
                observation gap, in chronological order. Used by CMC-corrected
                ORU. None for uncorrected ORU.

        Returns:
            Combined updated state.
        """
        self.history_obs.append(measurement)

        if measurement is None:
            if self._frozen_state is None:
                self._freeze()
            return self._get_combined_state()

        if self._frozen_state is not None:
            self._unfreeze(measurement, gap_affines)

        z = measurement.reshape(MEASUREMENT_DIM, 1)
        self._imm_update(z)

        return self._get_combined_state()

    def _imm_interact(self) -> None:
        """IMM interaction (mixing) step.

        Compute mixing probabilities and blend model states so each
        model starts its prediction from a weighted combination of
        all models' previous estimates.
        """
        num = self.num_models

        # Compute mixing probabilities: mu_ij = p(i->j) * prob(i) / c_j
        # c_j = sum_i p(i->j) * prob(i) = normalization constant
        c_bar = self.transition_prob.T @ self.model_probs
        # Prevent division by zero
        c_bar = np.maximum(c_bar, 1e-30)

        mu = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                mu[i, j] = (
                    self.transition_prob[i, j] * self.model_probs[i] / c_bar[j]
                )

        # Mix states in common subspace
        mixed_states = []
        mixed_covs = []

        for j in range(num):
            # Project all model states to common space
            x_common = [self._G[i] @ self.models[i].x for i in range(num)]

            # Mixed state for model j
            x_mixed = np.zeros((COMMON_DIM, 1))
            for i in range(num):
                x_mixed += mu[i, j] * x_common[i]

            # Mixed covariance for model j
            P_common = [
                self._G[i] @ self.models[i].P @ self._G[i].T for i in range(num)
            ]
            P_mixed = np.zeros((COMMON_DIM, COMMON_DIM))
            for i in range(num):
                diff = x_common[i] - x_mixed
                P_mixed += mu[i, j] * (P_common[i] + diff @ diff.T)

            mixed_states.append(x_mixed)
            mixed_covs.append(P_mixed)

        # Inject mixed states back into each model
        for j in range(num):
            model = self.models[j]
            # Save model-specific dimensions before overwriting
            x_specific = model.x.copy()

            # Inject common dimensions from mixed state
            model.x = self._G_plus[j] @ mixed_states[j]

            # Restore model-specific dimensions (acceleration, omega)
            # weighted by self-transition probability
            if model.dim > COMMON_DIM:
                for k in range(COMMON_DIM, model.dim):
                    model.x[k, 0] = x_specific[k, 0]

            # Inject covariance: common block from mix, specific block preserved
            P_new = np.eye(model.dim) * 1e-4
            P_injected = self._G_plus[j] @ mixed_covs[j] @ self._G_plus[j].T
            P_new[:COMMON_DIM, :COMMON_DIM] = P_injected[:COMMON_DIM, :COMMON_DIM]
            if model.dim > COMMON_DIM:
                for k in range(COMMON_DIM, model.dim):
                    P_new[k, k] = model.P[k, k]
            model.P = P_new

    def _imm_update(self, z: np.ndarray) -> None:
        """Run per-model updates and recompute model probabilities.

        Args:
            z: Measurement [cx, cy, w, h], shape (4, 1).
        """
        log_likelihoods = np.zeros(self.num_models)
        for j, model in enumerate(self.models):
            log_likelihoods[j] = model.update(z)

        # Update model probabilities
        c_bar = self.transition_prob.T @ self.model_probs

        # Use log-sum-exp trick for numerical stability
        log_probs = log_likelihoods + np.log(np.maximum(c_bar, 1e-30))
        max_log = np.max(log_probs)
        probs = np.exp(log_probs - max_log)
        total = probs.sum()

        if total > 0:
            self.model_probs = probs / total
        # else: keep previous probabilities (all likelihoods were -inf)

    def _get_combined_state(self) -> np.ndarray:
        """Compute probability-weighted combination of model states.

        Returns:
            Combined state in common subspace, shape (COMMON_DIM,).
        """
        x_combined = np.zeros((COMMON_DIM, 1))
        for j in range(self.num_models):
            x_common = self._G[j] @ self.models[j].x
            x_combined += self.model_probs[j] * x_common
        return x_combined[:, 0]

    def get_combined_covariance(self) -> np.ndarray:
        """Compute probability-weighted combined covariance in common subspace.

        Includes both the within-model covariance and the spread of
        model means (between-model covariance). This is the standard
        IMM combination formula for covariance.

        Returns:
            Combined covariance matrix, shape (COMMON_DIM, COMMON_DIM).
        """
        x_combined = np.zeros((COMMON_DIM, 1))
        for j in range(self.num_models):
            x_common = self._G[j] @ self.models[j].x
            x_combined += self.model_probs[j] * x_common

        P_combined = np.zeros((COMMON_DIM, COMMON_DIM))
        for j in range(self.num_models):
            x_common = self._G[j] @ self.models[j].x
            P_common = self._G[j] @ self.models[j].P @ self._G[j].T
            diff = x_common - x_combined
            P_combined += self.model_probs[j] * (P_common + diff @ diff.T)

        return P_combined

    def get_model_prob(self, name: str) -> float:
        """Get the current probability of a motion model by name.

        Args:
            name: Model name ("cv", "ca", or "ct").

        Returns:
            The model's current probability, or 0.0 if the model is not
            part of this filter's configuration.
        """
        try:
            idx = self.model_names.index(name)
        except ValueError:
            return 0.0
        return float(self.model_probs[idx])

    def get_gating_distribution(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the measurement-space distribution for Mahalanobis gating.

        Returns the combined predicted measurement mean and the innovation
        covariance S = P_meas + R, where P_meas is the measurement-space
        block of the IMM combined covariance (including between-model
        spread) and R is the size-dependent measurement noise evaluated
        at the combined state.

        Returns:
            Tuple of:
                mean: Predicted measurement [cx, cy, w, h], shape (4,).
                S: Innovation covariance, shape (4, 4).
        """
        mean = self._get_combined_state()[:MEASUREMENT_DIM]
        P_meas = self.get_combined_covariance()[
            :MEASUREMENT_DIM, :MEASUREMENT_DIM
        ]

        # Size-dependent measurement noise (same formula as the models)
        w = max(float(mean[2]), 1.0)
        h = max(float(mean[3]), 1.0)
        std = np.array([
            STD_WEIGHT_POSITION * w,
            STD_WEIGHT_POSITION * h,
            STD_WEIGHT_POSITION * w,
            STD_WEIGHT_POSITION * h,
        ])
        R = np.diag(std ** 2)

        return mean, P_meas + R

    def get_velocity(self) -> np.ndarray:
        """Get the combined-state center velocity.

        Returns:
            Velocity [v_cx, v_cy] in pixels per frame, shape (2,).
        """
        return self._get_combined_state()[4:6].copy()

    def get_state_xyxy(self) -> np.ndarray:
        """Get current combined estimate as [x1, y1, x2, y2] bbox.

        Returns:
            Bounding box in xyxy format, shape (4,).
        """
        state = self._get_combined_state()
        cxcywh = state[:4]
        # Ensure positive width/height
        cxcywh[2] = max(cxcywh[2], 1.0)
        cxcywh[3] = max(cxcywh[3], 1.0)
        return cxcywh_to_xyxy(cxcywh)

    def apply_affine(self, affine: np.ndarray) -> None:
        """Apply CMC affine transform to all model states.

        For [cx, cy, w, h] space:
        - (cx, cy): rotated + translated
        - (w, h): scaled by sqrt(det(R)) each (uniform scaling)
        - (v_cx, v_cy): rotated (no translation)
        - (v_w, v_h): scaled by sqrt(det(R))
        - Model-specific (a_cx, a_cy, omega): rotated or scaled accordingly

        Args:
            affine: 2x3 affine matrix from CMC.
        """
        R = affine[:2, :2]
        t = affine[:2, 2]
        det_R = np.linalg.det(R)
        scale = np.sqrt(abs(det_R))

        for model in self.models:
            dim = model.dim
            T = np.eye(dim)
            b = np.zeros((dim, 1))

            # Position: R @ [cx, cy] + t
            T[0:2, 0:2] = R
            b[0, 0] = t[0]
            b[1, 0] = t[1]

            # Width/height: scale by sqrt(det(R))
            T[2, 2] = scale
            T[3, 3] = scale

            # Velocity of position: R @ [v_cx, v_cy]
            T[4:6, 4:6] = R

            # Velocity of width/height: scale
            T[6, 6] = scale
            T[7, 7] = scale

            # Model-specific dimensions
            if isinstance(model, CAModel):
                # Acceleration: R @ [a_cx, a_cy]
                T[8:10, 8:10] = R
            elif isinstance(model, CTModel):
                # Omega (turn rate): invariant under affine
                T[8, 8] = 1.0

            model.x = T @ model.x + b
            model.P = T @ model.P @ T.T

    def _freeze(self) -> None:
        """Save IMM state for ORU re-update."""
        self._frozen_state = {
            "models": deepcopy([(m.x.copy(), m.P.copy()) for m in self.models]),
            "model_probs": self.model_probs.copy(),
            "history_obs": deepcopy(self.history_obs),
        }

    @staticmethod
    def _apply_affine_to_obs_cxcywh(
        obs: np.ndarray, affine: np.ndarray
    ) -> np.ndarray:
        """Apply 2x3 affine transform to an observation in [cx,cy,w,h] space.

        Args:
            obs: Observation as [cx, cy, w, h], shape (4,).
            affine: 2x3 affine matrix.

        Returns:
            Transformed observation, shape (4,).
        """
        R = affine[:2, :2]
        t = affine[:2, 2]
        scale = np.sqrt(abs(np.linalg.det(R)))
        cx, cy, w, h = obs
        new_center = R @ np.array([cx, cy]) + t
        return np.array([new_center[0], new_center[1], w * scale, h * scale])

    def _unfreeze(
        self,
        new_measurement: np.ndarray,
        gap_affines: list[np.ndarray] | None = None,
    ) -> None:
        """Restore frozen state and apply ORU virtual trajectory.

        When gap_affines are provided (CMC-corrected ORU), interpolation
        is done in the frozen frame's coordinate system, then each virtual
        observation is transformed forward using cumulative affines. The
        IMM state is also CMC-warped at each step.

        Args:
            new_measurement: New observation as [cx, cy, w, h], shape (4,).
            gap_affines: List of 2x3 affine matrices, one per gap frame,
                in chronological order. None for uncorrected ORU.
        """
        frozen = self._frozen_state
        if frozen is None:
            return

        # Restore model states
        for j, (x_saved, P_saved) in enumerate(frozen["models"]):
            self.models[j].x = x_saved
            self.models[j].P = P_saved
        self.model_probs = frozen["model_probs"]
        old_history = frozen["history_obs"]

        # Find last real observation before the gap
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

        gap_length = len(self.history_obs) - 1 - last_obs_idx

        if gap_length > 0:
            use_cmc = (
                gap_affines is not None and len(gap_affines) >= gap_length
            )

            if use_cmc:
                # CMC-corrected ORU
                cumul_3x3 = np.eye(3)
                for aff in gap_affines[:gap_length]:
                    aff_3x3 = np.eye(3)
                    aff_3x3[:2] = aff
                    cumul_3x3 = aff_3x3 @ cumul_3x3
                inv_cumul = np.linalg.inv(cumul_3x3)

                # Transform new_measurement to frozen frame
                new_obs_frozen = self._apply_affine_to_obs_cxcywh(
                    new_measurement, inv_cumul[:2]
                )

                partial_3x3 = np.eye(3)
                for step in range(1, gap_length + 1):
                    alpha = step / gap_length
                    virtual_frozen = (
                        last_obs * (1.0 - alpha) + new_obs_frozen * alpha
                    )

                    aff = gap_affines[step - 1]
                    aff_3x3 = np.eye(3)
                    aff_3x3[:2] = aff
                    partial_3x3 = aff_3x3 @ partial_3x3

                    virtual_obs = self._apply_affine_to_obs_cxcywh(
                        virtual_frozen, partial_3x3[:2]
                    )

                    self.predict()
                    self.apply_affine(aff)
                    z = virtual_obs.reshape(MEASUREMENT_DIM, 1)
                    self._imm_update(z)
            else:
                # Original ORU (no CMC correction)
                for step in range(1, gap_length + 1):
                    alpha = step / gap_length
                    virtual_obs = (
                        last_obs * (1.0 - alpha) + new_measurement * alpha
                    )
                    self.predict()
                    z = virtual_obs.reshape(MEASUREMENT_DIM, 1)
                    self._imm_update(z)

        self._frozen_state = None

    @property
    def model_probability_str(self) -> str:
        """Human-readable model probabilities for debugging."""
        parts = [
            f"{name}={prob:.2f}"
            for name, prob in zip(self.model_names, self.model_probs)
        ]
        return " ".join(parts)
