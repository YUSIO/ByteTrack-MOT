"""Tests for arcsort/track.py -- STrack lifecycle and state management."""

from __future__ import annotations

import numpy as np
import pytest

from arcsort.track import STrack, TrackState, reset_track_id


@pytest.fixture(autouse=True)
def _reset_ids():
    """Reset track ID counter before each test."""
    reset_track_id()
    yield
    reset_track_id()


def _make_bbox(x1: float = 10.0, y1: float = 20.0,
               x2: float = 50.0, y2: float = 80.0) -> np.ndarray:
    return np.array([x1, y1, x2, y2])


# ---------------------------------------------------------------------------
# Track creation
# ---------------------------------------------------------------------------

def test_track_creation_initial_state():
    track = STrack(_make_bbox(), confidence=0.9)
    assert track.state == TrackState.NEW
    assert track.confidence == 0.9
    assert track.hits == 1
    assert track.hit_streak == 1
    assert track.age == 1
    assert track.time_since_update == 0


def test_track_creation_assigns_unique_ids():
    t1 = STrack(_make_bbox(), confidence=0.9)
    t2 = STrack(_make_bbox(), confidence=0.8)
    assert t1.track_id != t2.track_id


def test_track_creation_sequential_ids():
    t1 = STrack(_make_bbox(), confidence=0.9)
    t2 = STrack(_make_bbox(), confidence=0.8)
    assert t2.track_id == t1.track_id + 1


def test_track_creation_stores_observation():
    bbox = _make_bbox()
    track = STrack(bbox, confidence=0.9)
    assert len(track.observations) == 1
    np.testing.assert_allclose(track.last_observation[:4], bbox)


def test_track_creation_velocity_zero():
    track = STrack(_make_bbox(), confidence=0.9)
    np.testing.assert_array_equal(track.velocity, [0.0, 0.0])


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def test_predict_advances_age():
    track = STrack(_make_bbox(), confidence=0.9)
    initial_age = track.age
    track.predict()
    assert track.age == initial_age + 1


def test_predict_increments_time_since_update():
    track = STrack(_make_bbox(), confidence=0.9)
    assert track.time_since_update == 0
    track.predict()
    assert track.time_since_update == 1


def test_predict_returns_bbox():
    track = STrack(_make_bbox(), confidence=0.9)
    bbox = track.predict()
    assert bbox.shape == (4,)
    assert np.all(np.isfinite(bbox))


def test_predict_resets_hit_streak_after_update_gap():
    track = STrack(_make_bbox(), confidence=0.9)
    track.predict()  # time_since_update becomes 1
    track.predict()  # time_since_update > 0, so hit_streak = 0
    assert track.hit_streak == 0


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

def test_update_resets_time_since_update():
    track = STrack(_make_bbox(), confidence=0.9)
    track.predict()
    assert track.time_since_update == 1
    track.update(_make_bbox(12.0, 22.0, 52.0, 82.0), confidence=0.85)
    assert track.time_since_update == 0


def test_update_increments_hits():
    track = STrack(_make_bbox(), confidence=0.9)
    initial_hits = track.hits
    track.predict()
    track.update(_make_bbox(12.0, 22.0, 52.0, 82.0), confidence=0.85)
    assert track.hits == initial_hits + 1


def test_update_increments_hit_streak():
    track = STrack(_make_bbox(), confidence=0.9)
    initial_streak = track.hit_streak
    track.predict()
    track.update(_make_bbox(12.0, 22.0, 52.0, 82.0), confidence=0.85)
    assert track.hit_streak == initial_streak + 1


def test_update_none_does_not_reset_time_since_update():
    track = STrack(_make_bbox(), confidence=0.9)
    track.predict()
    track.update(None)
    assert track.time_since_update == 1  # Not reset


def test_update_none_does_not_increment_hits():
    track = STrack(_make_bbox(), confidence=0.9)
    initial_hits = track.hits
    track.predict()
    track.update(None)
    assert track.hits == initial_hits


def test_update_stores_observation():
    track = STrack(_make_bbox(), confidence=0.9)
    track.predict()
    new_bbox = _make_bbox(15.0, 25.0, 55.0, 85.0)
    track.update(new_bbox, confidence=0.85)
    assert len(track.observations) == 2
    np.testing.assert_allclose(track.last_observation[:4], new_bbox)


# ---------------------------------------------------------------------------
# Lifecycle: NEW -> TRACKED -> LOST -> REMOVED
# ---------------------------------------------------------------------------

def test_lifecycle_new_to_tracked():
    track = STrack(_make_bbox(), confidence=0.9)
    assert track.state == TrackState.NEW
    track.predict()
    track.update(_make_bbox(12.0, 22.0, 52.0, 82.0), confidence=0.85)
    assert track.state == TrackState.TRACKED


def test_lifecycle_mark_lost():
    track = STrack(_make_bbox(), confidence=0.9)
    track.predict()
    track.update(_make_bbox(), confidence=0.9)
    track.mark_lost()
    assert track.state == TrackState.LOST


def test_lifecycle_mark_removed():
    track = STrack(_make_bbox(), confidence=0.9)
    track.mark_removed()
    assert track.state == TrackState.REMOVED


def test_lifecycle_lost_to_tracked():
    track = STrack(_make_bbox(), confidence=0.9)
    track.predict()
    track.update(_make_bbox(), confidence=0.9)
    track.mark_lost()
    assert track.state == TrackState.LOST
    track.predict()
    track.update(_make_bbox(12.0, 22.0, 52.0, 82.0), confidence=0.85)
    assert track.state == TrackState.TRACKED


# ---------------------------------------------------------------------------
# is_confirmed
# ---------------------------------------------------------------------------

def test_is_confirmed_new_track():
    track = STrack(_make_bbox(), confidence=0.9)
    assert not track.is_confirmed


def test_is_confirmed_tracked():
    track = STrack(_make_bbox(), confidence=0.9)
    track.predict()
    track.update(_make_bbox(), confidence=0.9)
    assert track.is_confirmed


# ---------------------------------------------------------------------------
# velocity computation
# ---------------------------------------------------------------------------

def test_velocity_after_observations():
    """After update with prev obs available, velocity should be non-zero."""
    track = STrack(_make_bbox(0.0, 0.0, 10.0, 10.0), confidence=0.9)
    track.predict()
    # Move the box to the right
    track.update(_make_bbox(10.0, 0.0, 20.0, 10.0), confidence=0.9)
    # After delta_t more frames, velocity should be computed
    for i in range(3):
        track.predict()
        track.update(
            _make_bbox(20.0 + (i + 1) * 10.0, 0.0,
                       30.0 + (i + 1) * 10.0, 10.0),
            confidence=0.9,
        )
    # Velocity should indicate rightward movement (dx > 0)
    assert track.velocity[1] > 0  # dx component


# ---------------------------------------------------------------------------
# get_state / get_previous_obs
# ---------------------------------------------------------------------------

def test_get_state_returns_xyxy():
    track = STrack(_make_bbox(), confidence=0.9)
    bbox = track.get_state()
    assert bbox.shape == (4,)
    assert np.all(np.isfinite(bbox))


def test_get_previous_obs_returns_shape():
    track = STrack(_make_bbox(), confidence=0.9)
    prev = track.get_previous_obs()
    assert prev.shape == (5,)


# ---------------------------------------------------------------------------
# reset_track_id
# ---------------------------------------------------------------------------

def test_reset_track_id_resets_counter():
    t1 = STrack(_make_bbox(), confidence=0.9)
    assert t1.track_id == 0
    reset_track_id()
    t2 = STrack(_make_bbox(), confidence=0.9)
    assert t2.track_id == 0
