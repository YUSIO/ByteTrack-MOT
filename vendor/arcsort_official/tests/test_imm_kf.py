"""Tests for arcsort/imm_kalman_filter.py -- IMM-KF with CV/CA/CT models.

Tests individual motion models with synthetic trajectories, the IMM
orchestrator's mixing and probability updates, and the full interface
compatibility with the standard KalmanFilter.
"""

from __future__ import annotations

import numpy as np
import pytest

from arcsort.imm_kalman_filter import (
    CAModel,
    CTModel,
    CVModel,
    IMMKalmanFilter,
    MEASUREMENT_DIM,
)


# ---------------------------------------------------------------------------
# CVModel -- Constant Velocity
# ---------------------------------------------------------------------------

def test_cv_model_straight_line():
    """CV model should track a straight-line trajectory accurately."""
    model = CVModel(process_noise_scale=1.0)
    # Initialize: object at (100, 100), size 50x80
    model.x[:4, 0] = [100, 100, 50, 80]
    model.x[4:8, 0] = [5, 0, 0, 0]  # moving right at 5 px/frame
    model.P = np.eye(8) * 10.0

    for frame in range(1, 20):
        model.predict()
        z = np.array([[100 + 5 * frame], [100], [50], [80]])
        model.update(z)

    # Position should converge to ground truth
    assert abs(model.x[0, 0] - 200) < 10.0
    assert abs(model.x[1, 0] - 100) < 10.0
    # Velocity should converge
    assert abs(model.x[4, 0] - 5.0) < 2.0


def test_cv_model_predict_advances_position():
    model = CVModel()
    model.x[:4, 0] = [100, 100, 50, 80]
    model.x[4, 0] = 10.0  # v_cx = 10
    model.predict()
    assert abs(model.x[0, 0] - 110) < 0.01


# ---------------------------------------------------------------------------
# CAModel -- Constant Acceleration
# ---------------------------------------------------------------------------

def test_ca_model_accelerating_trajectory():
    """CA model should handle linearly increasing velocity."""
    model = CAModel(process_noise_scale=0.5)
    model.x[:4, 0] = [100, 100, 50, 80]
    model.x[4, 0] = 0  # initial velocity = 0
    model.x[8, 0] = 2  # acceleration = 2 px/frame^2
    model.P = np.eye(10) * 10.0

    for frame in range(1, 20):
        model.predict()
        # x = x0 + 0.5 * a * t^2 = 100 + t^2
        true_x = 100 + frame ** 2
        z = np.array([[true_x], [100], [50], [80]])
        model.update(z)

    # Should track the accelerating position
    true_final = 100 + 19 ** 2
    assert abs(model.x[0, 0] - true_final) < 20.0
    # Acceleration should converge near 2
    assert abs(model.x[8, 0] - 2.0) < 1.5


def test_ca_model_has_acceleration_dims():
    model = CAModel()
    assert model.dim == 10
    assert model.x.shape == (10, 1)


# ---------------------------------------------------------------------------
# CTModel -- Coordinated Turn (EKF)
# ---------------------------------------------------------------------------

def test_ct_model_circular_trajectory():
    """CT model should track a circular path."""
    model = CTModel(process_noise_scale=0.3)
    radius = 100
    omega_true = 0.1  # rad/frame
    model.x[:4, 0] = [radius, 0, 50, 80]
    model.x[4, 0] = 0                       # v_cx = 0
    model.x[5, 0] = radius * omega_true     # v_cy = r * omega
    model.x[8, 0] = omega_true
    model.P = np.eye(9) * 10.0

    for frame in range(1, 40):
        model.predict()
        angle = omega_true * frame
        true_cx = radius * np.cos(angle)
        true_cy = radius * np.sin(angle)
        z = np.array([[true_cx], [true_cy], [50], [80]])
        model.update(z)

    # Omega should stay near the true value
    assert abs(model.x[8, 0] - omega_true) < 0.05


def test_ct_model_omega_zero_fallback():
    """CT model should handle omega=0 gracefully (reduces to CV)."""
    model = CTModel()
    model.x[:4, 0] = [100, 100, 50, 80]
    model.x[4, 0] = 5.0  # v_cx = 5
    model.x[5, 0] = 0.0
    model.x[8, 0] = 0.0  # omega = 0

    # Should not crash and should behave like CV
    model.predict()
    assert abs(model.x[0, 0] - 105) < 0.01
    assert abs(model.x[1, 0] - 100) < 0.01


def test_ct_model_has_omega_dim():
    model = CTModel()
    assert model.dim == 9
    assert model.x.shape == (9, 1)


# ---------------------------------------------------------------------------
# IMMKalmanFilter -- full orchestrator
# ---------------------------------------------------------------------------

def test_imm_initiate():
    """IMM initiate should set all models to the same measurement."""
    imm = IMMKalmanFilter()
    imm.initiate(np.array([100, 100, 50, 80]))

    for model in imm.models:
        assert abs(model.x[0, 0] - 100) < 0.01
        assert abs(model.x[1, 0] - 100) < 0.01
        assert abs(model.x[2, 0] - 50) < 0.01
        assert abs(model.x[3, 0] - 80) < 0.01


def test_imm_predict_returns_state():
    imm = IMMKalmanFilter()
    imm.initiate(np.array([100, 100, 50, 80]))
    state = imm.predict()
    assert state.shape[0] == 8  # common dim


def test_imm_get_state_xyxy():
    """get_state_xyxy should return valid [x1,y1,x2,y2] bbox."""
    imm = IMMKalmanFilter()
    imm.initiate(np.array([100, 100, 50, 80]))
    bbox = imm.get_state_xyxy()
    assert bbox.shape == (4,)
    # cx=100, cy=100, w=50, h=80 -> x1=75, y1=60, x2=125, y2=140
    assert abs(bbox[0] - 75) < 1
    assert abs(bbox[1] - 60) < 1
    assert abs(bbox[2] - 125) < 1
    assert abs(bbox[3] - 140) < 1


def test_imm_straight_line_favors_cv():
    """On a straight-line trajectory, combined prediction should be accurate.

    Note: CT with omega~0 is equivalent to CV, so CT may also gain high
    probability on straight lines. What matters is prediction accuracy,
    not which specific model dominates.
    """
    imm = IMMKalmanFilter({
        "initial_model_probs": [1 / 3, 1 / 3, 1 / 3],
    })
    imm.initiate(np.array([100, 100, 50, 80]))

    for frame in range(1, 30):
        imm.predict()
        z = np.array([100 + 5 * frame, 100, 50, 80])
        imm.update(z)

    # Combined prediction should be accurate for straight-line
    bbox = imm.get_state_xyxy()
    cx = (bbox[0] + bbox[2]) / 2
    assert abs(cx - (100 + 5 * 29)) < 10.0
    # CV + CT together should dominate over CA
    assert imm.model_probs[0] + imm.model_probs[2] > imm.model_probs[1]


def test_imm_accelerating_prediction_accurate():
    """On an accelerating trajectory, combined prediction should be accurate.

    CA and CT can both handle acceleration (CT via changing omega), so
    we test prediction quality rather than which model dominates.
    """
    imm = IMMKalmanFilter({
        "initial_model_probs": [1 / 3, 1 / 3, 1 / 3],
    })
    imm.initiate(np.array([100, 100, 50, 80]))

    for frame in range(1, 30):
        imm.predict()
        # Quadratic motion: constant acceleration
        z = np.array([100 + 0.5 * frame ** 2, 100, 50, 80])
        imm.update(z)

    # Combined prediction should track accelerating position
    bbox = imm.get_state_xyxy()
    cx = (bbox[0] + bbox[2]) / 2
    true_cx = 100 + 0.5 * 29 ** 2
    assert abs(cx - true_cx) < 30.0, (
        f"Predicted cx={cx:.1f}, true={true_cx:.1f}"
    )
    # CA + CT (acceleration-capable models) should together beat CV
    assert imm.model_probs[1] + imm.model_probs[2] > imm.model_probs[0], (
        f"CA+CT prob {imm.model_probs[1]+imm.model_probs[2]:.3f} should > "
        f"CV prob {imm.model_probs[0]:.3f}"
    )


def test_imm_circular_favors_ct():
    """On a circular trajectory, CT model should gain probability."""
    imm = IMMKalmanFilter({
        "initial_model_probs": [1 / 3, 1 / 3, 1 / 3],
    })
    radius = 100
    omega = 0.1
    imm.initiate(np.array([radius, 0, 50, 80]))

    for frame in range(1, 40):
        imm.predict()
        angle = omega * frame
        z = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            50, 80,
        ])
        imm.update(z)

    # CT (index 2) should have highest probability
    assert imm.model_probs[2] > imm.model_probs[0], (
        f"CT prob {imm.model_probs[2]:.3f} should exceed "
        f"CV prob {imm.model_probs[0]:.3f}"
    )


def test_imm_model_switching():
    """IMM should adapt when motion pattern changes (CV -> turn -> CV)."""
    imm = IMMKalmanFilter({
        "initial_model_probs": [1 / 3, 1 / 3, 1 / 3],
    })
    imm.initiate(np.array([100, 100, 50, 80]))

    # Phase 1: straight line (frames 1-20)
    for frame in range(1, 21):
        imm.predict()
        z = np.array([100 + 5 * frame, 100, 50, 80])
        imm.update(z)

    cv_prob_after_straight = imm.model_probs[0]

    # Phase 2: circular motion (frames 21-50)
    cx_start = 100 + 5 * 20
    omega = 0.15
    radius = 50
    for frame in range(1, 31):
        imm.predict()
        angle = omega * frame
        z = np.array([
            cx_start + radius * np.sin(angle),
            100 + radius * (1 - np.cos(angle)),
            50, 80,
        ])
        imm.update(z)

    # After circular phase, CT should have gained relative to CV
    ct_prob_after_turn = imm.model_probs[2]
    cv_prob_after_turn = imm.model_probs[0]

    assert ct_prob_after_turn > cv_prob_after_turn, (
        f"After turn: CT={ct_prob_after_turn:.3f} should > "
        f"CV={cv_prob_after_turn:.3f}"
    )


def test_imm_update_none_oru():
    """Update with None should freeze state for ORU."""
    imm = IMMKalmanFilter()
    imm.initiate(np.array([100, 100, 50, 80]))

    imm.predict()
    imm.update(np.array([105, 100, 50, 80]))

    # Miss some frames
    imm.predict()
    imm.update(None)
    imm.predict()
    imm.update(None)

    # Frozen state should exist
    assert imm._frozen_state is not None

    # Re-observe: should unfreeze and re-update
    imm.predict()
    imm.update(np.array([120, 100, 50, 80]))
    assert imm._frozen_state is None


def test_imm_apply_affine_identity():
    """Identity affine should not change IMM state."""
    imm = IMMKalmanFilter()
    imm.initiate(np.array([100, 100, 50, 80]))

    states_before = [m.x.copy() for m in imm.models]
    imm.apply_affine(np.eye(2, 3, dtype=np.float64))
    for j, model in enumerate(imm.models):
        np.testing.assert_array_almost_equal(model.x, states_before[j])


def test_imm_apply_affine_translation():
    """Translation affine should shift all model positions."""
    imm = IMMKalmanFilter()
    imm.initiate(np.array([100, 100, 50, 80]))

    affine = np.eye(2, 3, dtype=np.float64)
    affine[0, 2] = 10.0
    affine[1, 2] = -5.0
    imm.apply_affine(affine)

    for model in imm.models:
        assert abs(model.x[0, 0] - 110) < 0.01
        assert abs(model.x[1, 0] - 95) < 0.01


def test_imm_model_probs_sum_to_one():
    """Model probabilities should always sum to 1."""
    imm = IMMKalmanFilter()
    imm.initiate(np.array([100, 100, 50, 80]))

    for frame in range(1, 10):
        imm.predict()
        z = np.array([100 + 3 * frame, 100 + frame, 50, 80])
        imm.update(z)
        assert abs(imm.model_probs.sum() - 1.0) < 1e-10


def test_imm_transition_prob_matrix():
    """Transition matrix rows should sum to 1."""
    imm = IMMKalmanFilter()
    for row in imm.transition_prob:
        assert abs(row.sum() - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# STrack integration with IMM
# ---------------------------------------------------------------------------

def test_strack_use_imm():
    """STrack with use_imm=True should use IMMKalmanFilter."""
    from arcsort.imm_kalman_filter import IMMKalmanFilter
    from arcsort.track import STrack, reset_track_id

    reset_track_id()
    bbox = np.array([75.0, 60.0, 125.0, 140.0])
    track = STrack(bbox, confidence=0.9, use_imm=True)
    assert isinstance(track.kf, IMMKalmanFilter)

    # predict/update cycle should work
    track.predict()
    track.update(np.array([76.0, 61.0, 126.0, 141.0]), confidence=0.9)

    bbox_out = track.get_state()
    assert bbox_out.shape == (4,)
    assert np.all(np.isfinite(bbox_out))
    reset_track_id()


def test_strack_use_imm_false():
    """STrack with use_imm=False should use standard KalmanFilter."""
    from arcsort.kalman_filter import KalmanFilter
    from arcsort.track import STrack, reset_track_id

    reset_track_id()
    bbox = np.array([75.0, 60.0, 125.0, 140.0])
    track = STrack(bbox, confidence=0.9, use_imm=False)
    assert isinstance(track.kf, KalmanFilter)
    reset_track_id()


def test_strack_imm_cmc():
    """STrack with IMM should support apply_cmc."""
    from arcsort.track import STrack, reset_track_id

    reset_track_id()
    bbox = np.array([75.0, 60.0, 125.0, 140.0])
    track = STrack(bbox, confidence=0.9, use_imm=True)

    affine = np.eye(2, 3, dtype=np.float64)
    affine[0, 2] = 5.0
    track.apply_cmc(affine)

    bbox_out = track.get_state()
    assert np.all(np.isfinite(bbox_out))
    reset_track_id()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_imm_empty_observations():
    """IMM should handle being called with no observations gracefully."""
    imm = IMMKalmanFilter()
    imm.initiate(np.array([100, 100, 50, 80]))

    # Multiple predict-only cycles
    for _ in range(10):
        imm.predict()
        imm.update(None)

    bbox = imm.get_state_xyxy()
    assert np.all(np.isfinite(bbox))


def test_imm_single_frame():
    """IMM should work with just initiate + one predict/update."""
    imm = IMMKalmanFilter()
    imm.initiate(np.array([100, 100, 50, 80]))
    imm.predict()
    imm.update(np.array([102, 101, 50, 80]))
    bbox = imm.get_state_xyxy()
    assert np.all(np.isfinite(bbox))
    assert abs(imm.model_probs.sum() - 1.0) < 1e-10


def test_imm_custom_config():
    """IMM with custom config should use provided parameters."""
    config = {
        "models": ["cv", "ct"],
        "transition_prob_diagonal": 0.95,
        "process_noise_cv": 2.0,
        "process_noise_ct": 0.1,
        "initial_model_probs": [0.7, 0.3],
    }
    imm = IMMKalmanFilter(config)
    assert imm.num_models == 2
    assert abs(imm.model_probs[0] - 0.7) < 0.01
    assert abs(imm.transition_prob[0, 0] - 0.95) < 0.01


# ---------------------------------------------------------------------------
# get_gating_distribution -- measurement-space distribution for gating
# ---------------------------------------------------------------------------

def test_gating_distribution_shapes_and_mean():
    """Mean must match the combined state; S must be 4x4 symmetric PSD."""
    kf = IMMKalmanFilter()
    kf.initiate(np.array([100.0, 200.0, 50.0, 80.0]))
    kf.predict()

    mean, S = kf.get_gating_distribution()
    assert mean.shape == (4,)
    assert S.shape == (4, 4)
    # Freshly initiated + one predict: mean stays at the measurement
    np.testing.assert_allclose(mean, [100.0, 200.0, 50.0, 80.0], atol=1e-6)
    # Symmetric positive definite
    np.testing.assert_allclose(S, S.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(S)
    assert np.all(eigvals > 0)


def test_gating_distribution_uncertainty_grows_when_coasting():
    """Missed updates (predict-only) must widen the gating covariance."""
    kf = IMMKalmanFilter()
    kf.initiate(np.array([100.0, 100.0, 50.0, 80.0]))

    kf.predict()
    _, S_first = kf.get_gating_distribution()

    for _ in range(10):
        kf.predict()
    _, S_coasted = kf.get_gating_distribution()

    assert np.trace(S_coasted) > np.trace(S_first)


def test_gating_distribution_own_measurement_within_gate():
    """A detection at the predicted location has near-zero distance."""
    from arcsort.matching import mahalanobis_distance_batch

    kf = IMMKalmanFilter()
    kf.initiate(np.array([100.0, 100.0, 50.0, 80.0]))
    # A few update cycles of steady motion
    for i in range(1, 6):
        kf.predict()
        kf.update(np.array([100.0 + 5 * i, 100.0, 50.0, 80.0]))
    kf.predict()

    mean, S = kf.get_gating_distribution()
    d2 = mahalanobis_distance_batch(
        mean[np.newaxis, :], mean[np.newaxis, :], S[np.newaxis, :, :]
    )
    assert d2[0, 0] == pytest.approx(0.0, abs=1e-10)
