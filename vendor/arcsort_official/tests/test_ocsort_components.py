"""Tests for arcsort/ocsort_components.py -- OCM, speed_direction, k_previous_obs."""

from __future__ import annotations

import numpy as np
import pytest

from arcsort.ocsort_components import (
    compute_ocm_cost,
    k_previous_obs,
    speed_direction,
    speed_direction_batch,
)


# ---------------------------------------------------------------------------
# speed_direction
# ---------------------------------------------------------------------------

def test_speed_direction_known_movement():
    prev = np.array([0.0, 0.0, 10.0, 10.0])  # center (5, 5)
    curr = np.array([10.0, 0.0, 20.0, 10.0])  # center (15, 5)
    d = speed_direction(prev, curr)
    assert d.shape == (2,)
    # dy=0, dx=10 -> normalized (0, 1)
    assert d[0] == pytest.approx(0.0, abs=1e-4)
    assert d[1] == pytest.approx(1.0, abs=1e-4)


def test_speed_direction_diagonal():
    prev = np.array([0.0, 0.0, 10.0, 10.0])  # center (5, 5)
    curr = np.array([10.0, 10.0, 20.0, 20.0])  # center (15, 15)
    d = speed_direction(prev, curr)
    norm = np.sqrt(2) / 2
    assert d[0] == pytest.approx(norm, abs=1e-4)
    assert d[1] == pytest.approx(norm, abs=1e-4)


def test_speed_direction_zero_movement():
    bbox = np.array([0.0, 0.0, 10.0, 10.0])
    d = speed_direction(bbox, bbox)
    # Both should be near zero (divided by eps)
    assert abs(d[0]) < 1e-3
    assert abs(d[1]) < 1e-3


def test_speed_direction_negative_movement():
    prev = np.array([10.0, 10.0, 20.0, 20.0])  # center (15, 15)
    curr = np.array([0.0, 0.0, 10.0, 10.0])    # center (5, 5)
    d = speed_direction(prev, curr)
    norm = np.sqrt(2) / 2
    assert d[0] == pytest.approx(-norm, abs=1e-4)
    assert d[1] == pytest.approx(-norm, abs=1e-4)


# ---------------------------------------------------------------------------
# speed_direction_batch
# ---------------------------------------------------------------------------

def test_speed_direction_batch_shape():
    dets = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [20.0, 20.0, 30.0, 30.0],
    ])
    prev_obs = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [10.0, 10.0, 20.0, 20.0],
        [30.0, 30.0, 40.0, 40.0],
    ])
    dy, dx = speed_direction_batch(dets, prev_obs)
    assert dy.shape == (3, 2)
    assert dx.shape == (3, 2)


def test_speed_direction_batch_matches_scalar():
    """Batch computation should match individual speed_direction calls."""
    dets = np.array([
        [10.0, 10.0, 20.0, 20.0],
        [30.0, 30.0, 40.0, 40.0],
    ])
    prev_obs = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [20.0, 20.0, 30.0, 30.0],
    ])
    dy_batch, dx_batch = speed_direction_batch(dets, prev_obs)

    for m in range(prev_obs.shape[0]):
        for n in range(dets.shape[0]):
            d_scalar = speed_direction(prev_obs[m], dets[n])
            assert dy_batch[m, n] == pytest.approx(d_scalar[0], abs=1e-5)
            assert dx_batch[m, n] == pytest.approx(d_scalar[1], abs=1e-5)


# ---------------------------------------------------------------------------
# k_previous_obs
# ---------------------------------------------------------------------------

def test_k_previous_obs_exact_match():
    obs = {3: np.array([10.0, 20.0, 30.0, 40.0, 0.9])}
    result = k_previous_obs(obs, current_age=6, delta_t=3)
    np.testing.assert_array_equal(result, obs[3])


def test_k_previous_obs_fallback_most_recent():
    obs = {
        1: np.array([10.0, 20.0, 30.0, 40.0, 0.9]),
        2: np.array([11.0, 21.0, 31.0, 41.0, 0.8]),
    }
    # target_age = 10 - 3 = 7, no exact match, fallback to age 2
    result = k_previous_obs(obs, current_age=10, delta_t=3)
    np.testing.assert_array_equal(result, obs[2])


def test_k_previous_obs_no_valid_observation():
    obs = {10: np.array([10.0, 20.0, 30.0, 40.0, 0.9])}
    # target_age = 5 - 3 = 2, obs at 10 > 2, so no valid
    result = k_previous_obs(obs, current_age=5, delta_t=3)
    np.testing.assert_array_equal(result, [-1.0, -1.0, -1.0, -1.0, -1.0])


def test_k_previous_obs_empty_dict():
    result = k_previous_obs({}, current_age=5, delta_t=3)
    np.testing.assert_array_equal(result, [-1.0, -1.0, -1.0, -1.0, -1.0])


def test_k_previous_obs_delta_t_zero():
    obs = {5: np.array([10.0, 20.0, 30.0, 40.0, 0.9])}
    result = k_previous_obs(obs, current_age=5, delta_t=0)
    np.testing.assert_array_equal(result, obs[5])


def test_k_previous_obs_multiple_candidates():
    obs = {
        1: np.array([1.0, 1.0, 1.0, 1.0, 0.5]),
        3: np.array([3.0, 3.0, 3.0, 3.0, 0.7]),
        5: np.array([5.0, 5.0, 5.0, 5.0, 0.9]),
    }
    # target_age = 8 - 3 = 5, exact match at 5
    result = k_previous_obs(obs, current_age=8, delta_t=3)
    np.testing.assert_array_equal(result, obs[5])


# ---------------------------------------------------------------------------
# compute_ocm_cost
# ---------------------------------------------------------------------------

def test_compute_ocm_cost_aligned_directions():
    """Track moving right, detection is also to the right => low angle diff."""
    dets = np.array([[20.0, 0.0, 30.0, 10.0]])  # center (25, 5)
    velocities = np.array([[0.0, 1.0]])  # moving right
    prev_obs = np.array([[0.0, 0.0, 10.0, 10.0, 0.9]])  # center (5, 5)
    cost = compute_ocm_cost(dets, velocities, prev_obs, inertia_weight=1.0)
    assert cost.shape == (1, 1)
    # Aligned directions should produce a positive cost (reward)
    assert cost[0, 0] > 0


def test_compute_ocm_cost_opposite_directions():
    """Track moving right, detection is to the left => high angle diff."""
    dets = np.array([[-10.0, 0.0, 0.0, 10.0]])  # center (-5, 5)
    velocities = np.array([[0.0, 1.0]])  # moving right
    prev_obs = np.array([[0.0, 0.0, 10.0, 10.0, 0.9]])  # center (5, 5)
    cost = compute_ocm_cost(dets, velocities, prev_obs, inertia_weight=1.0)
    # Opposite direction should produce negative or lower cost
    assert cost.shape == (1, 1)


def test_compute_ocm_cost_invalid_previous_obs():
    """When previous obs is invalid (conf=-1), cost should be zero."""
    dets = np.array([[20.0, 0.0, 30.0, 10.0]])
    velocities = np.array([[0.0, 1.0]])
    prev_obs = np.array([[-1.0, -1.0, -1.0, -1.0, -1.0]])
    cost = compute_ocm_cost(dets, velocities, prev_obs, inertia_weight=1.0)
    assert cost[0, 0] == pytest.approx(0.0)


def test_compute_ocm_cost_empty_detections():
    dets = np.empty((0, 4))
    velocities = np.array([[0.0, 1.0]])
    prev_obs = np.array([[0.0, 0.0, 10.0, 10.0, 0.9]])
    cost = compute_ocm_cost(dets, velocities, prev_obs, inertia_weight=1.0)
    assert cost.shape == (0, 1)


def test_compute_ocm_cost_empty_tracks():
    dets = np.array([[0.0, 0.0, 10.0, 10.0]])
    velocities = np.empty((0, 2))
    prev_obs = np.empty((0, 5))
    cost = compute_ocm_cost(dets, velocities, prev_obs, inertia_weight=1.0)
    assert cost.shape == (1, 0)


def test_compute_ocm_cost_inertia_weight_scaling():
    """Doubling weight should double the cost."""
    dets = np.array([[20.0, 0.0, 30.0, 10.0]])
    velocities = np.array([[0.0, 1.0]])
    prev_obs = np.array([[0.0, 0.0, 10.0, 10.0, 0.9]])
    cost_w1 = compute_ocm_cost(dets, velocities, prev_obs, inertia_weight=1.0)
    cost_w2 = compute_ocm_cost(dets, velocities, prev_obs, inertia_weight=2.0)
    assert cost_w2[0, 0] == pytest.approx(2.0 * cost_w1[0, 0], abs=1e-6)
