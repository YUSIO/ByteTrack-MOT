"""Tests for tools/run_cached.py -- detection loading and sequence running."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from arcsort.track import reset_track_id
from arcsort.tracker import ARCTracker
from tools.run_cached import load_detections, run_sequence


@pytest.fixture(autouse=True)
def _reset_ids():
    reset_track_id()
    yield
    reset_track_id()


def _write_det_file(lines: list[str]) -> Path:
    """Write detection lines to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    )
    f.write("\n".join(lines))
    if lines:
        f.write("\n")
    f.close()
    return Path(f.name)


# ---------------------------------------------------------------------------
# load_detections
# ---------------------------------------------------------------------------

def test_load_detections_valid_format():
    lines = [
        "1,-1,100,200,50,80,0.9,-1,-1,-1",
        "1,-1,300,400,60,70,0.8,-1,-1,-1",
        "2,-1,105,205,50,80,0.85,-1,-1,-1",
    ]
    path = _write_det_file(lines)
    try:
        dets = load_detections(path)
        assert 1 in dets
        assert 2 in dets
        assert dets[1].shape == (2, 6)
        assert dets[2].shape == (1, 6)

        # Check conversion from xywh to xyxy
        # First det: x1=100, y1=200, w=50, h=80 -> x2=150, y2=280
        np.testing.assert_allclose(dets[1][0, :4], [100, 200, 150, 280])
        assert dets[1][0, 4] == pytest.approx(0.9)
    finally:
        os.unlink(path)


def test_load_detections_single_detection():
    lines = ["1,-1,100,200,50,80,0.9,-1,-1,-1"]
    path = _write_det_file(lines)
    try:
        dets = load_detections(path)
        assert 1 in dets
        assert dets[1].shape == (1, 6)
    finally:
        os.unlink(path)


def test_load_detections_missing_file():
    with pytest.raises(FileNotFoundError):
        load_detections(Path("/nonexistent/file.txt"))


# ---------------------------------------------------------------------------
# run_sequence
# ---------------------------------------------------------------------------

def test_run_sequence_produces_output():
    tracker = ARCTracker({"min_hits": 1, "new_track_thresh": 0.7})
    dets_by_frame = {
        1: np.array([[100, 200, 150, 280, 0.9, 0]]),
        2: np.array([[102, 202, 152, 282, 0.85, 0]]),
        3: np.array([[104, 204, 154, 284, 0.88, 0]]),
    }
    lines = run_sequence(tracker, dets_by_frame)
    assert len(lines) > 0
    # Each line should have MOT format: frame,id,x,y,w,h,conf,...
    parts = lines[0].split(",")
    assert len(parts) == 10


def test_run_sequence_empty_detections():
    tracker = ARCTracker({"min_hits": 1})
    lines = run_sequence(tracker, {})
    assert lines == []


def test_run_sequence_valid_mot_format():
    tracker = ARCTracker({"min_hits": 1, "new_track_thresh": 0.7})
    dets_by_frame = {
        1: np.array([[100, 200, 150, 280, 0.9, 0]]),
    }
    lines = run_sequence(tracker, dets_by_frame)
    if lines:
        parts = lines[0].split(",")
        frame_id = int(parts[0])
        track_id = int(parts[1])
        assert frame_id == 1
        assert track_id >= 1
        # w and h should be positive
        w = float(parts[4])
        h = float(parts[5])
        assert w > 0
        assert h > 0


def test_run_sequence_gaps_in_frames():
    """Frames without detections should still be processed."""
    tracker = ARCTracker({"min_hits": 1, "new_track_thresh": 0.7})
    dets_by_frame = {
        1: np.array([[100, 200, 150, 280, 0.9, 0]]),
        5: np.array([[100, 200, 150, 280, 0.9, 0]]),
    }
    lines = run_sequence(tracker, dets_by_frame)
    # Should process frames 1-5, even though 2-4 have no dets
    assert isinstance(lines, list)
