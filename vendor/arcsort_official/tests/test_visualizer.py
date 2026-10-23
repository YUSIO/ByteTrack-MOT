"""Tests for arcsort/visualizer.py."""

from __future__ import annotations

import numpy as np
import pytest

from arcsort.visualizer import Visualizer

pytest.importorskip("supervision")


def _pan_sequence(n: int = 6, dx: float = 10.0):
    """A static object seen by a camera panning so the scene shifts dx px/frame."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    step = np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0]])
    for t in range(n):
        x = 50 + dx * t
        yield frame, np.array([[x, 100, x + 20, 140, 1]]), (step if t > 0 else None)


def test_plain_trails_unchanged():
    viz = Visualizer(trace_length=10)
    for frame, tracks, _ in _pan_sequence():
        out = viz.annotate(frame, tracks)
    assert out.shape == (240, 320, 3)


def test_motion_compensated_trail_fixed_to_scene():
    pytest.importorskip("trackers")
    viz = Visualizer(trace_length=10, motion_compensated_trails=True)
    for frame, tracks, motion in _pan_sequence():
        viz.annotate(frame, tracks, camera_motion=motion)
    trail = np.array(viz.trace_annotator._trajectories[1])
    assert len(trail) == 6
    # The object never moved in the scene, so every world point is the same
    np.testing.assert_allclose(trail, np.broadcast_to(trail[0], trail.shape), atol=1e-6)


def test_reset_clears_trails_and_motion():
    pytest.importorskip("trackers")
    viz = Visualizer(trace_length=10, motion_compensated_trails=True)
    for frame, tracks, motion in _pan_sequence():
        viz.annotate(frame, tracks, camera_motion=motion)
    viz.reset()
    np.testing.assert_array_equal(viz._world_to_frame, np.eye(3))
    assert not viz.trace_annotator._trajectories
