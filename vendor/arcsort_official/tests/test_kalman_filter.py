"""Tests for arcsort/kalman_filter.py -- Kalman filter with ORU support."""

from __future__ import annotations

import numpy as np
import pytest

from arcsort.kalman_filter import KalmanFilter, MEASUREMENT_DIM, STATE_DIM


def _make_measurement(cx: float, cy: float, s: float, r: float) -> np.ndarray:
    return np.array([cx, cy, s, r])


# ---------------------------------------------------------------------------
# initiate
# ---------------------------------------------------------------------------

def test_initiate_sets_state():
    kf = KalmanFilter()
    m = _make_measurement(100.0, 200.0, 5000.0, 0.5)
    kf.initiate(m)
    assert kf.x[0, 0] == pytest.approx(100.0)
    assert kf.x[1, 0] == pytest.approx(200.0)
    assert kf.x[2, 0] == pytest.approx(5000.0)
    assert kf.x[3, 0] == pytest.approx(0.5)


def test_initiate_velocities_zero():
    kf = KalmanFilter()
    m = _make_measurement(100.0, 200.0, 5000.0, 0.5)
    kf.initiate(m)
    assert kf.x[4, 0] == pytest.approx(0.0)
    assert kf.x[5, 0] == pytest.approx(0.0)
    assert kf.x[6, 0] == pytest.approx(0.0)


def test_initiate_state_dimensions():
    kf = KalmanFilter()
    m = _make_measurement(50.0, 50.0, 1000.0, 1.0)
    kf.initiate(m)
    assert kf.x.shape == (STATE_DIM, 1)
    assert kf.P.shape == (STATE_DIM, STATE_DIM)


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def test_predict_returns_correct_shape():
    kf = KalmanFilter()
    kf.initiate(_make_measurement(100.0, 200.0, 5000.0, 0.5))
    state = kf.predict()
    assert state.shape == (STATE_DIM,)


def test_predict_advances_state_with_velocity():
    kf = KalmanFilter()
    kf.initiate(_make_measurement(100.0, 200.0, 5000.0, 0.5))
    # Set velocity manually
    kf.x[4, 0] = 10.0   # v_cx
    kf.x[5, 0] = -5.0   # v_cy
    kf.x[6, 0] = 100.0  # v_s
    state = kf.predict()
    assert state[0] == pytest.approx(110.0)  # cx + v_cx
    assert state[1] == pytest.approx(195.0)  # cy + v_cy
    assert state[2] == pytest.approx(5100.0)  # s + v_s


def test_predict_clamps_negative_scale_velocity():
    """If s + v_s <= 0, the velocity should be zeroed before predict."""
    kf = KalmanFilter()
    kf.initiate(_make_measurement(100.0, 200.0, 50.0, 0.5))
    kf.x[6, 0] = -100.0  # v_s that would make s negative
    state = kf.predict()
    # After clamping v_s to 0, s stays at 50
    assert state[2] == pytest.approx(50.0)


def test_predict_does_not_clamp_positive_scale_velocity():
    kf = KalmanFilter()
    kf.initiate(_make_measurement(100.0, 200.0, 5000.0, 0.5))
    kf.x[6, 0] = 50.0
    state = kf.predict()
    assert state[2] == pytest.approx(5050.0)


def test_predict_covariance_grows():
    kf = KalmanFilter()
    kf.initiate(_make_measurement(100.0, 200.0, 5000.0, 0.5))
    p_before = np.trace(kf.P)
    kf.predict()
    p_after = np.trace(kf.P)
    # Process noise Q is added, so covariance trace should grow
    assert p_after > p_before


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

def test_update_with_measurement_improves_estimate():
    kf = KalmanFilter()
    kf.initiate(_make_measurement(100.0, 200.0, 5000.0, 0.5))
    kf.predict()
    p_before_update = np.trace(kf.P)
    kf.update(_make_measurement(102.0, 198.0, 5010.0, 0.5))
    p_after_update = np.trace(kf.P)
    assert p_after_update < p_before_update


def test_update_none_does_not_crash():
    kf = KalmanFilter()
    kf.initiate(_make_measurement(100.0, 200.0, 5000.0, 0.5))
    kf.predict()
    state = kf.update(None)
    assert state.shape == (STATE_DIM,)


def test_update_none_triggers_freeze():
    kf = KalmanFilter()
    kf.initiate(_make_measurement(100.0, 200.0, 5000.0, 0.5))
    kf.predict()
    assert kf._frozen_state is None
    kf.update(None)
    assert kf._frozen_state is not None


def test_update_none_does_not_change_state():
    kf = KalmanFilter()
    kf.initiate(_make_measurement(100.0, 200.0, 5000.0, 0.5))
    kf.predict()
    x_before = kf.x.copy()
    kf.update(None)
    np.testing.assert_array_equal(kf.x, x_before)


def test_update_returns_correct_shape():
    kf = KalmanFilter()
    kf.initiate(_make_measurement(100.0, 200.0, 5000.0, 0.5))
    kf.predict()
    state = kf.update(_make_measurement(100.0, 200.0, 5000.0, 0.5))
    assert state.shape == (STATE_DIM,)


# ---------------------------------------------------------------------------
# ORU: freeze / unfreeze
# ---------------------------------------------------------------------------

def test_oru_freeze_and_unfreeze():
    """After gap then re-observation, frozen_state should be cleared."""
    kf = KalmanFilter()
    kf.initiate(_make_measurement(100.0, 200.0, 5000.0, 0.5))
    kf.predict()
    kf.update(_make_measurement(102.0, 202.0, 5050.0, 0.5))

    # Now simulate gap: predict + update(None) for 3 frames
    for _ in range(3):
        kf.predict()
        kf.update(None)

    assert kf._frozen_state is not None

    # Re-observe
    kf.predict()
    kf.update(_make_measurement(120.0, 220.0, 5200.0, 0.5))

    assert kf._frozen_state is None


def test_oru_multiple_none_updates_keep_frozen():
    kf = KalmanFilter()
    kf.initiate(_make_measurement(100.0, 200.0, 5000.0, 0.5))
    kf.predict()
    kf.update(_make_measurement(102.0, 202.0, 5050.0, 0.5))

    # Multiple None updates
    for _ in range(5):
        kf.predict()
        kf.update(None)

    assert kf._frozen_state is not None


def test_oru_unfreeze_produces_reasonable_state():
    """After ORU re-update, state should be closer to the new measurement
    than it would be with just prediction drift."""
    kf = KalmanFilter()
    m_init = _make_measurement(100.0, 200.0, 5000.0, 0.5)
    kf.initiate(m_init)
    kf.predict()
    kf.update(_make_measurement(102.0, 202.0, 5050.0, 0.5))

    # Gap of 5 frames
    for _ in range(5):
        kf.predict()
        kf.update(None)

    # Re-observe at a shifted position
    m_new = _make_measurement(130.0, 230.0, 5500.0, 0.5)
    kf.predict()
    kf.update(m_new)

    # State should be near the new measurement, not wildly off
    assert abs(kf.x[0, 0] - 130.0) < 20.0
    assert abs(kf.x[1, 0] - 230.0) < 20.0


def test_oru_history_obs_grows():
    kf = KalmanFilter()
    kf.initiate(_make_measurement(100.0, 200.0, 5000.0, 0.5))
    for i in range(5):
        kf.predict()
        kf.update(_make_measurement(100.0 + i, 200.0, 5000.0, 0.5))
    assert len(kf.history_obs) == 5


# ---------------------------------------------------------------------------
# predict/update convergence
# ---------------------------------------------------------------------------

def test_multiple_predict_update_converges():
    """Repeated predict/update with same measurement should converge."""
    kf = KalmanFilter()
    m = _make_measurement(100.0, 200.0, 5000.0, 0.5)
    kf.initiate(m)

    for _ in range(50):
        kf.predict()
        kf.update(m)

    assert kf.x[0, 0] == pytest.approx(100.0, abs=1.0)
    assert kf.x[1, 0] == pytest.approx(200.0, abs=1.0)
    assert kf.x[2, 0] == pytest.approx(5000.0, abs=50.0)
    # Velocities should converge near zero
    assert abs(kf.x[4, 0]) < 1.0
    assert abs(kf.x[5, 0]) < 1.0


def test_predict_update_with_moving_target():
    """Filter should track a linearly moving target."""
    kf = KalmanFilter()
    kf.initiate(_make_measurement(100.0, 200.0, 5000.0, 0.5))

    for i in range(1, 30):
        kf.predict()
        kf.update(_make_measurement(100.0 + i * 2.0, 200.0 + i * 1.0, 5000.0, 0.5))

    # After 30 frames, filter should be near the latest measurement
    assert abs(kf.x[0, 0] - (100.0 + 29 * 2.0)) < 5.0
    assert abs(kf.x[1, 0] - (200.0 + 29 * 1.0)) < 5.0


# ---------------------------------------------------------------------------
# get_state_xyxy
# ---------------------------------------------------------------------------

def test_get_state_xyxy_shape():
    kf = KalmanFilter()
    kf.initiate(_make_measurement(100.0, 200.0, 5000.0, 0.5))
    bbox = kf.get_state_xyxy()
    assert bbox.shape == (4,)


def test_get_state_xyxy_consistent_with_initiation():
    """get_state_xyxy should round-trip reasonably from original xyxy."""
    from arcsort.utils import xyxy_to_cxcysr
    original_xyxy = np.array([10.0, 20.0, 50.0, 80.0])
    m = xyxy_to_cxcysr(original_xyxy)
    kf = KalmanFilter()
    kf.initiate(m)
    recovered = kf.get_state_xyxy()
    np.testing.assert_allclose(recovered, original_xyxy, atol=0.5)
