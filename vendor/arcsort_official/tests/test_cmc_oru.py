"""Tests for CMC-corrected ORU.

Tests that the ORU virtual trajectory is properly warped when CMC affine
transforms are provided during the observation gap. Covers both the
standard KF and the IMM-KF.
"""

from __future__ import annotations

import numpy as np
import pytest

from arcsort.kalman_filter import KalmanFilter
from arcsort.imm_kalman_filter import IMMKalmanFilter
from arcsort.track import STrack, reset_track_id
from arcsort.tracker import ARCTracker
from arcsort.utils import xyxy_to_cxcysr, xyxy_to_cxcywh


@pytest.fixture(autouse=True)
def _reset_ids():
    reset_track_id()
    yield
    reset_track_id()


# ---------------------------------------------------------------------------
# KalmanFilter: CMC-corrected ORU
# ---------------------------------------------------------------------------

def test_kf_unfreeze_no_affines_unchanged():
    """Without gap_affines, ORU should behave exactly as before."""
    kf = KalmanFilter()
    obs = xyxy_to_cxcysr(np.array([100.0, 100.0, 200.0, 200.0]))
    kf.initiate(obs)

    # Observe, then lose, then re-observe
    kf.predict()
    kf.update(obs)
    kf.predict()
    kf.update(None)  # freeze
    kf.predict()
    kf.update(None)
    kf.predict()
    state = kf.update(obs)  # unfreeze with no affines

    bbox = kf.get_state_xyxy()
    assert np.all(np.isfinite(bbox))


def test_kf_unfreeze_identity_affines_unchanged():
    """Identity affines should produce same result as no affines."""
    # Run without affines
    kf1 = KalmanFilter()
    obs = xyxy_to_cxcysr(np.array([100.0, 100.0, 200.0, 200.0]))
    kf1.initiate(obs)
    kf1.predict()
    kf1.update(obs)
    kf1.predict()
    kf1.update(None)
    kf1.predict()
    kf1.update(None)
    kf1.predict()
    kf1.update(obs)
    state1 = kf1.get_state_xyxy()

    # Run with identity affines
    kf2 = KalmanFilter()
    kf2.initiate(obs)
    kf2.predict()
    kf2.update(obs)
    kf2.predict()
    kf2.update(None)
    kf2.predict()
    kf2.update(None)
    kf2.predict()
    identity_affines = [np.eye(2, 3), np.eye(2, 3), np.eye(2, 3)]
    kf2.update(obs, gap_affines=identity_affines)
    state2 = kf2.get_state_xyxy()

    np.testing.assert_allclose(state1, state2, atol=1e-4)


def test_kf_unfreeze_translation_affine():
    """Translation affine should shift virtual observations correctly."""
    kf = KalmanFilter()
    obs1 = xyxy_to_cxcysr(np.array([100.0, 100.0, 200.0, 200.0]))
    kf.initiate(obs1)
    kf.predict()
    kf.update(obs1)

    # Freeze
    kf.predict()
    kf.update(None)

    # Re-observe at shifted position (as if camera panned right by 50px)
    obs2 = xyxy_to_cxcysr(np.array([150.0, 100.0, 250.0, 200.0]))

    # Affine: pure translation of 50px right
    affine = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 0.0]])
    kf.predict()
    kf.update(obs2, gap_affines=[affine, affine])

    bbox = kf.get_state_xyxy()
    assert np.all(np.isfinite(bbox))
    # The state should be near obs2 after CMC correction
    assert bbox[0] > 100  # should have shifted right


# ---------------------------------------------------------------------------
# IMMKalmanFilter: CMC-corrected ORU
# ---------------------------------------------------------------------------

def test_imm_unfreeze_no_affines_unchanged():
    """IMM ORU without affines should work as before."""
    kf = IMMKalmanFilter()
    obs = xyxy_to_cxcywh(np.array([100.0, 100.0, 200.0, 200.0]))
    kf.initiate(obs)

    kf.predict()
    kf.update(obs)
    kf.predict()
    kf.update(None)
    kf.predict()
    kf.update(None)
    kf.predict()
    kf.update(obs)

    bbox = kf.get_state_xyxy()
    assert np.all(np.isfinite(bbox))


def test_imm_unfreeze_identity_affines():
    """IMM: identity affines should match no-affine result."""
    obs = xyxy_to_cxcywh(np.array([100.0, 100.0, 200.0, 200.0]))

    # Without affines
    kf1 = IMMKalmanFilter()
    kf1.initiate(obs)
    kf1.predict()
    kf1.update(obs)
    kf1.predict()
    kf1.update(None)
    kf1.predict()
    kf1.update(None)
    kf1.predict()
    kf1.update(obs)
    state1 = kf1.get_state_xyxy()

    # With identity affines
    kf2 = IMMKalmanFilter()
    kf2.initiate(obs)
    kf2.predict()
    kf2.update(obs)
    kf2.predict()
    kf2.update(None)
    kf2.predict()
    kf2.update(None)
    kf2.predict()
    identity_affines = [np.eye(2, 3), np.eye(2, 3), np.eye(2, 3)]
    kf2.update(obs, gap_affines=identity_affines)
    state2 = kf2.get_state_xyxy()

    np.testing.assert_allclose(state1, state2, atol=1e-4)


def test_imm_unfreeze_translation():
    """IMM: translation affine should produce valid state."""
    kf = IMMKalmanFilter()
    obs1 = xyxy_to_cxcywh(np.array([100.0, 100.0, 200.0, 200.0]))
    kf.initiate(obs1)
    kf.predict()
    kf.update(obs1)

    kf.predict()
    kf.update(None)

    obs2 = xyxy_to_cxcywh(np.array([150.0, 100.0, 250.0, 200.0]))
    affine = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 0.0]])
    kf.predict()
    kf.update(obs2, gap_affines=[affine, affine])

    bbox = kf.get_state_xyxy()
    assert np.all(np.isfinite(bbox))


# ---------------------------------------------------------------------------
# STrack: gap_affines threading
# ---------------------------------------------------------------------------

def test_strack_update_accepts_gap_affines():
    """STrack.update should accept and pass gap_affines without error."""
    track = STrack(np.array([100, 100, 200, 200], dtype=float), 0.9)
    track.predict()
    track.update(np.array([101, 101, 201, 201], dtype=float), 0.9)
    track.predict()
    track.update(None)
    track.predict()

    affines = [np.eye(2, 3), np.eye(2, 3)]
    track.update(
        np.array([102, 102, 202, 202], dtype=float), 0.9,
        gap_affines=affines,
    )
    bbox = track.get_state()
    assert np.all(np.isfinite(bbox))


def test_strack_update_none_ignores_gap_affines():
    """Passing gap_affines with None bbox should not error."""
    track = STrack(np.array([100, 100, 200, 200], dtype=float), 0.9)
    track.predict()
    # None update should ignore gap_affines gracefully
    track.update(None, gap_affines=[np.eye(2, 3)])


# ---------------------------------------------------------------------------
# ARCTracker: _gather_gap_affines
# ---------------------------------------------------------------------------

def test_tracker_gather_gap_affines_no_cmc():
    """Without CMC, _gather_gap_affines should return None."""
    tracker = ARCTracker({"use_cmc": False, "min_hits": 1})
    track = STrack(np.array([100, 100, 200, 200], dtype=float), 0.9)
    track.time_since_update = 3
    result = tracker._gather_gap_affines(track)
    assert result is None


def test_tracker_gather_gap_affines_no_gap():
    """Track with time_since_update=0 should return None."""
    tracker = ARCTracker({"use_cmc": True, "min_hits": 1})
    track = STrack(np.array([100, 100, 200, 200], dtype=float), 0.9)
    track.time_since_update = 0
    result = tracker._gather_gap_affines(track)
    assert result is None


def test_tracker_gather_gap_affines_with_stored():
    """Should gather stored affines for the gap period when camera moved."""
    tracker = ARCTracker({"use_cmc": True, "min_hits": 1})
    tracker.frame_count = 10

    # Store affines with significant translation (> 1px)
    for f in range(1, 11):
        aff = np.eye(2, 3)
        aff[0, 2] = f * 5.0  # 5px translation per frame
        tracker.affine_transforms[f] = aff

    track = STrack(np.array([100, 100, 200, 200], dtype=float), 0.9)
    track.time_since_update = 3

    result = tracker._gather_gap_affines(track)
    assert result is not None
    assert len(result) == 3  # frames 8, 9, 10
    # Verify it got the right frames
    np.testing.assert_allclose(result[0][0, 2], 40.0)  # frame 8: 5*8
    np.testing.assert_allclose(result[1][0, 2], 45.0)  # frame 9: 5*9
    np.testing.assert_allclose(result[2][0, 2], 50.0)  # frame 10: 5*10


def test_tracker_gather_gap_affines_missing_frames():
    """Missing affines should be filled with identity when camera moved."""
    tracker = ARCTracker({"use_cmc": True, "min_hits": 1})
    tracker.frame_count = 5

    # Store affine with significant translation for frame 4
    tracker.affine_transforms[4] = np.array([[1.0, 0, 20.0], [0, 1.0, 0]])

    track = STrack(np.array([100, 100, 200, 200], dtype=float), 0.9)
    track.time_since_update = 3

    result = tracker._gather_gap_affines(track)
    assert result is not None
    assert len(result) == 3
    # Frame 3: missing -> identity
    np.testing.assert_array_equal(result[0], np.eye(2, 3))
    # Frame 4: stored (20px translation)
    np.testing.assert_allclose(result[1][0, 2], 20.0)
    # Frame 5: missing -> identity
    np.testing.assert_array_equal(result[2], np.eye(2, 3))


def test_tracker_gather_gap_affines_static_camera():
    """Static camera (near-identity affines) should return None."""
    tracker = ARCTracker({"use_cmc": True, "min_hits": 1})
    tracker.frame_count = 5

    # Store near-identity affines (< 1px translation)
    for f in range(3, 6):
        tracker.affine_transforms[f] = np.array(
            [[1.0, 0.0001, 0.1], [0.0001, 1.0, 0.05]]
        )

    track = STrack(np.array([100, 100, 200, 200], dtype=float), 0.9)
    track.time_since_update = 3

    result = tracker._gather_gap_affines(track)
    # Static camera -> should return None to avoid noise
    assert result is None


# ---------------------------------------------------------------------------
# End-to-end: tracker with CMC + ORU
# ---------------------------------------------------------------------------

def _make_checkerboard_frame(h: int = 480, w: int = 640) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    block = 40
    for y in range(0, h, block):
        for x in range(0, w, block):
            if ((x // block) + (y // block)) % 2 == 0:
                frame[y:y + block, x:x + block] = 255
    return frame


def _det(x1, y1, x2, y2, conf=0.9, cls=0.0):
    return np.array([x1, y1, x2, y2, conf, cls])


def test_tracker_cmc_oru_e2e():
    """End-to-end: CMC-corrected ORU should not crash or regress."""
    tracker = ARCTracker({
        "min_hits": 1,
        "new_track_thresh": 0.7,
        "track_buffer": 10,
        "use_cmc": True,
    })
    frame = _make_checkerboard_frame()

    # Establish track
    for _ in range(3):
        dets = np.array([_det(100, 100, 200, 200)])
        tracker.update(dets, frame=frame)

    track_id = tracker.tracks[0].track_id

    # Lose for 2 frames (CMC still runs, affines stored)
    tracker.update(np.empty((0, 6)), frame=frame)
    tracker.update(np.empty((0, 6)), frame=frame)

    # Re-appear -- should trigger CMC-corrected ORU
    dets = np.array([_det(101, 101, 201, 201)])
    tracker.update(dets, frame=frame)

    # Track should survive
    active = [t for t in tracker.tracks if t.time_since_update == 0]
    assert len(active) > 0
