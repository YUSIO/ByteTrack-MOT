"""Tests for arcsort/cmc.py -- Camera Motion Compensation."""

from __future__ import annotations

import numpy as np
import pytest

from arcsort.cmc import CameraMotionCompensator, DEFAULT_CMC_CONFIG


# ---------------------------------------------------------------------------
# CameraMotionCompensator initialization
# ---------------------------------------------------------------------------

def test_cmc_init_default_config():
    cmc = CameraMotionCompensator()
    assert cmc.downscale == DEFAULT_CMC_CONFIG["downscale"]
    assert cmc.min_inliers == DEFAULT_CMC_CONFIG["min_inliers"]


def test_cmc_init_custom_config():
    cmc = CameraMotionCompensator({"downscale": 4, "min_inliers": 20})
    assert cmc.downscale == 4
    assert cmc.min_inliers == 20


# ---------------------------------------------------------------------------
# First frame returns identity
# ---------------------------------------------------------------------------

def test_cmc_first_frame_returns_identity():
    cmc = CameraMotionCompensator()
    frame = _make_random_frame(480, 640)
    affine = cmc.apply(frame)
    expected = np.eye(2, 3, dtype=np.float64)
    np.testing.assert_array_almost_equal(affine, expected)


# ---------------------------------------------------------------------------
# Static camera produces near-identity
# ---------------------------------------------------------------------------

def test_cmc_static_camera_near_identity():
    """Two identical frames should produce a near-identity transform."""
    cmc = CameraMotionCompensator({"downscale": 1})
    frame = _make_textured_frame(480, 640)

    cmc.apply(frame)
    affine = cmc.apply(frame)

    # Should be very close to identity
    identity = np.eye(2, 3, dtype=np.float64)
    np.testing.assert_allclose(affine, identity, atol=0.1)


# ---------------------------------------------------------------------------
# Translated frame produces correct offset
# ---------------------------------------------------------------------------

def test_cmc_detects_horizontal_translation():
    """Camera panning right should produce positive tx."""
    cmc = CameraMotionCompensator({"downscale": 1, "min_inliers": 4})
    frame1 = _make_textured_frame(480, 640)

    # Shift frame right by 10 pixels
    frame2 = np.zeros_like(frame1)
    shift = 10
    frame2[:, shift:, :] = frame1[:, :640 - shift, :]

    cmc.apply(frame1)
    affine = cmc.apply(frame2)

    # tx should be approximately +10 (camera moved right)
    assert affine[0, 2] > 5.0, f"Expected positive tx, got {affine[0, 2]}"


def test_cmc_detects_vertical_translation():
    """Camera tilting down should produce positive ty."""
    cmc = CameraMotionCompensator({"downscale": 1, "min_inliers": 4})
    frame1 = _make_textured_frame(480, 640)

    # Shift frame down by 10 pixels
    frame2 = np.zeros_like(frame1)
    shift = 10
    frame2[shift:, :, :] = frame1[:480 - shift, :, :]

    cmc.apply(frame1)
    affine = cmc.apply(frame2)

    # ty should be approximately +10
    assert affine[1, 2] > 5.0, f"Expected positive ty, got {affine[1, 2]}"


# ---------------------------------------------------------------------------
# Reset clears state
# ---------------------------------------------------------------------------

def test_cmc_reset():
    cmc = CameraMotionCompensator()
    frame = _make_textured_frame(480, 640)
    cmc.apply(frame)
    assert cmc._prev_frame is not None

    cmc.reset()
    assert cmc._prev_frame is None
    assert cmc._prev_keypoints is None


# ---------------------------------------------------------------------------
# Graceful fallback with no features
# ---------------------------------------------------------------------------

def test_cmc_featureless_first_frame_returns_identity():
    """A featureless first frame should produce identity (no keypoints)."""
    cmc = CameraMotionCompensator({"downscale": 1})

    # First frame: uniform (no corners detected)
    frame1 = np.full((480, 640, 3), 128, dtype=np.uint8)
    affine1 = cmc.apply(frame1)

    identity = np.eye(2, 3, dtype=np.float64)
    np.testing.assert_array_almost_equal(affine1, identity)

    # Second frame: also uniform -- should still return identity
    frame2 = np.full((480, 640, 3), 100, dtype=np.uint8)
    affine2 = cmc.apply(frame2)
    np.testing.assert_array_almost_equal(affine2, identity)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_cmc_output_shape():
    cmc = CameraMotionCompensator()
    frame = _make_random_frame(480, 640)
    affine = cmc.apply(frame)
    assert affine.shape == (2, 3)


# ---------------------------------------------------------------------------
# Downscale handling
# ---------------------------------------------------------------------------

def test_cmc_downscale_produces_correct_translation():
    """Downscaled processing should still produce full-resolution translation."""
    cmc = CameraMotionCompensator({"downscale": 2, "min_inliers": 4})
    frame1 = _make_textured_frame(480, 640)

    shift = 20
    frame2 = np.zeros_like(frame1)
    frame2[:, shift:, :] = frame1[:, :640 - shift, :]

    cmc.apply(frame1)
    affine = cmc.apply(frame2)

    # Translation should be in full resolution (~20), not half (~10)
    assert affine[0, 2] > 10.0, (
        f"Expected full-res translation >10, got {affine[0, 2]}"
    )


# ---------------------------------------------------------------------------
# KF apply_affine
# ---------------------------------------------------------------------------

def test_kf_apply_affine_identity():
    """Identity affine should not change KF state."""
    from arcsort.kalman_filter import KalmanFilter

    kf = KalmanFilter()
    kf.initiate(np.array([100.0, 200.0, 5000.0, 0.5]))
    state_before = kf.x.copy()
    P_before = kf.P.copy()

    kf.apply_affine(np.eye(2, 3, dtype=np.float64))

    np.testing.assert_array_almost_equal(kf.x, state_before)
    np.testing.assert_array_almost_equal(kf.P, P_before)


def test_kf_apply_affine_translation():
    """Pure translation should only shift cx, cy."""
    from arcsort.kalman_filter import KalmanFilter

    kf = KalmanFilter()
    kf.initiate(np.array([100.0, 200.0, 5000.0, 0.5]))

    affine = np.eye(2, 3, dtype=np.float64)
    affine[0, 2] = 10.0  # tx
    affine[1, 2] = -5.0  # ty

    kf.apply_affine(affine)

    assert abs(kf.x[0, 0] - 110.0) < 1e-6
    assert abs(kf.x[1, 0] - 195.0) < 1e-6
    assert abs(kf.x[2, 0] - 5000.0) < 1e-6  # area unchanged
    assert abs(kf.x[3, 0] - 0.5) < 1e-6     # ratio unchanged


def test_kf_apply_affine_scale():
    """Uniform scale should scale area by scale^2."""
    from arcsort.kalman_filter import KalmanFilter

    kf = KalmanFilter()
    kf.initiate(np.array([100.0, 100.0, 1000.0, 0.5]))

    scale = 1.1
    affine = np.array([
        [scale, 0.0, 0.0],
        [0.0, scale, 0.0],
    ])

    kf.apply_affine(affine)

    assert abs(kf.x[0, 0] - 110.0) < 1e-6       # cx * 1.1
    assert abs(kf.x[1, 0] - 110.0) < 1e-6       # cy * 1.1
    assert abs(kf.x[2, 0] - 1210.0) < 1e-6      # area * 1.1^2
    assert abs(kf.x[3, 0] - 0.5) < 1e-6         # ratio unchanged


def test_kf_apply_affine_rotation():
    """90-degree rotation should swap cx/cy and rotate velocities."""
    from arcsort.kalman_filter import KalmanFilter

    kf = KalmanFilter()
    kf.initiate(np.array([100.0, 0.0, 1000.0, 0.5]))
    # Set velocity: moving right at 5 px/frame
    kf.x[4, 0] = 5.0  # v_cx
    kf.x[5, 0] = 0.0  # v_cy

    # 90-degree CCW rotation
    theta = np.pi / 2
    affine = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
    ])

    kf.apply_affine(affine)

    # After 90 CCW: (100, 0) -> (0, 100)
    assert abs(kf.x[0, 0] - 0.0) < 1e-6
    assert abs(kf.x[1, 0] - 100.0) < 1e-6
    # Velocity (5, 0) -> (0, 5)
    assert abs(kf.x[4, 0] - 0.0) < 1e-6
    assert abs(kf.x[5, 0] - 5.0) < 1e-6
    # Area and ratio should be preserved (det(R) = 1 for pure rotation)
    assert abs(kf.x[2, 0] - 1000.0) < 1e-6
    assert abs(kf.x[3, 0] - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# STrack.apply_cmc
# ---------------------------------------------------------------------------

def test_strack_apply_cmc():
    """STrack.apply_cmc should delegate to KF.apply_affine."""
    from arcsort.track import STrack, reset_track_id

    reset_track_id()
    bbox = np.array([100.0, 100.0, 200.0, 200.0])
    track = STrack(bbox, confidence=0.9)

    state_before = track.kf.x.copy()

    affine = np.eye(2, 3, dtype=np.float64)
    affine[0, 2] = 15.0
    track.apply_cmc(affine)

    # cx should have shifted by 15
    assert abs(track.kf.x[0, 0] - (state_before[0, 0] + 15.0)) < 1e-6
    reset_track_id()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_random_frame(h: int, w: int) -> np.ndarray:
    """Create a random BGR frame with texture."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_textured_frame(h: int, w: int) -> np.ndarray:
    """Create a frame with strong corner features (checkerboard-like)."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    block = 40
    for y in range(0, h, block):
        for x in range(0, w, block):
            if ((x // block) + (y // block)) % 2 == 0:
                frame[y:y + block, x:x + block] = 255
    return frame
