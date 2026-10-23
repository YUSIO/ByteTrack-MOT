"""Tests for the public API: presets, from_preset, supervision interop."""

from __future__ import annotations

import numpy as np
import pytest

from arcsort import ARCTracker
from arcsort.configs import available, load_preset


def test_shipped_presets():
    assert {"sportsmot", "dancetrack", "sportsmot_deepeiou_inputs"} <= set(available())


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        load_preset("nope")


def test_from_preset_without_weights_is_motion_only():
    tracker = ARCTracker.from_preset("sportsmot")
    assert not tracker.use_reid
    assert tracker.embedding_computer is None
    assert tracker.use_imm and tracker.use_cmc


def test_from_preset_overrides_merge_sub_configs():
    tracker = ARCTracker.from_preset(
        "sportsmot", track_buffer=60, eiou={"adaptive_scale": 3.0}
    )
    assert tracker.track_buffer == 60
    assert tracker.cfg["eiou"]["adaptive_scale"] == 3.0
    # untouched sub-config keys keep the preset values
    assert tracker.cfg["eiou"]["max_expansion"] == 1.2


def test_from_preset_tracks_a_moving_box():
    tracker = ARCTracker.from_preset("sportsmot", use_cmc=False)
    ids = set()
    for t in range(10):
        x = 100.0 + 5 * t
        dets = np.array([[x, 100.0, x + 50, 200.0, 0.9, 0]])
        out = tracker.update(dets)
        assert out.shape[1] == 5
        ids.update(out[:, 4].astype(int).tolist())
    assert len(ids) == 1


def test_supervision_roundtrip():
    sv = pytest.importorskip("supervision")
    from arcsort.interop import from_supervision, to_supervision

    dets = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10], [5, 5, 20, 20]], dtype=np.float32),
        confidence=np.array([0.9, 0.4]),
        class_id=np.array([0, 2]),
    )
    arr = from_supervision(dets)
    assert arr.shape == (2, 6)
    np.testing.assert_allclose(arr[:, 4], [0.9, 0.4])
    np.testing.assert_array_equal(arr[:, 5], [0, 2])
    assert from_supervision(sv.Detections.empty()).shape == (0, 6)

    tracked = to_supervision(np.array([[0, 0, 10, 10, 7]]))
    assert tracked.tracker_id.tolist() == [7]
    assert len(to_supervision(np.empty((0, 5)))) == 0


def test_independent_trackers_have_independent_ids():
    a = ARCTracker.from_preset("sportsmot", use_cmc=False)
    b = ARCTracker.from_preset("sportsmot", use_cmc=False)
    det = np.array([[100.0, 100.0, 150.0, 200.0, 0.9, 0]])
    for _ in range(3):
        ids_a = a.update(det)[:, 4].tolist()
        ids_b = b.update(det)[:, 4].tolist()
    assert ids_a == ids_b == [1.0]
    # resetting one tracker must not affect the other's numbering
    a.reset()
    two = np.array([[100.0, 100.0, 150.0, 200.0, 0.9, 0], [400.0, 100.0, 450.0, 200.0, 0.9, 0]])
    for _ in range(3):
        ids_b = b.update(two)[:, 4].tolist()
    assert sorted(ids_b) == [1.0, 2.0]
