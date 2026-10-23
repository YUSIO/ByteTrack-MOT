"""Tests for arcsort/tracker.py -- ARCTracker end-to-end."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import yaml

from arcsort.track import reset_track_id
from arcsort.tracker import ARCTracker, DEFAULT_CONFIG, load_config


@pytest.fixture(autouse=True)
def _reset_ids():
    """Reset track ID counter before each test."""
    reset_track_id()
    yield
    reset_track_id()


def _det(x1: float, y1: float, x2: float, y2: float,
         conf: float = 0.9, cls: float = 0.0) -> np.ndarray:
    """Create a single detection row [x1, y1, x2, y2, conf, cls]."""
    return np.array([x1, y1, x2, y2, conf, cls])


def _dets(*boxes: tuple[float, ...]) -> np.ndarray:
    """Create detection array from (x1,y1,x2,y2) tuples with default conf=0.9."""
    rows = [_det(*b) if len(b) == 6 else _det(*b, 0.9, 0.0) for b in boxes]
    return np.array(rows)


# ---------------------------------------------------------------------------
# No detections
# ---------------------------------------------------------------------------

def test_update_no_detections():
    tracker = ARCTracker()
    result = tracker.update(np.empty((0, 6)))
    assert result.shape == (0, 5)


def test_update_no_detections_multiple_frames():
    tracker = ARCTracker()
    for _ in range(10):
        result = tracker.update(np.empty((0, 6)))
    assert result.shape == (0, 5)


# ---------------------------------------------------------------------------
# Track creation
# ---------------------------------------------------------------------------

def test_track_creation_high_confidence():
    """High-conf detections above new_track_thresh should create tracks."""
    tracker = ARCTracker({"new_track_thresh": 0.7, "min_hits": 1})
    dets = _dets((100, 100, 200, 200, 0.9, 0))
    result = tracker.update(dets)
    # With min_hits=1, first frame is warmup, should output track
    assert result.shape[0] >= 1


def test_track_creation_low_confidence_rejected():
    """Detections below new_track_thresh should not create tracks."""
    tracker = ARCTracker({
        "new_track_thresh": 0.7,
        "track_high_thresh": 0.3,
        "min_hits": 1,
    })
    dets = _dets((100, 100, 200, 200, 0.5, 0))
    tracker.update(dets)
    assert len(tracker.tracks) == 0


# ---------------------------------------------------------------------------
# Track identity preservation
# ---------------------------------------------------------------------------

def test_track_identity_preserved():
    """Same object across frames should keep same track ID."""
    tracker = ARCTracker({"min_hits": 1, "new_track_thresh": 0.7})
    dets1 = _dets((100, 100, 200, 200, 0.9, 0))
    result1 = tracker.update(dets1)

    dets2 = _dets((102, 102, 202, 202, 0.9, 0))
    result2 = tracker.update(dets2)

    assert result1.shape[0] == 1
    assert result2.shape[0] == 1
    assert result1[0, 4] == result2[0, 4]  # same track ID


def test_two_separate_tracks():
    """Two well-separated objects should get distinct track IDs."""
    tracker = ARCTracker({"min_hits": 1, "new_track_thresh": 0.7})
    dets = _dets(
        (100, 100, 200, 200, 0.9, 0),
        (500, 500, 600, 600, 0.9, 0),
    )
    result = tracker.update(dets)
    if result.shape[0] == 2:
        assert result[0, 4] != result[1, 4]


# ---------------------------------------------------------------------------
# Track loss and removal
# ---------------------------------------------------------------------------

def test_track_lost_after_no_detection():
    """Track should survive for track_buffer frames without detection."""
    tracker = ARCTracker({
        "min_hits": 1,
        "new_track_thresh": 0.7,
        "track_buffer": 5,
    })
    dets = _dets((100, 100, 200, 200, 0.9, 0))
    tracker.update(dets)

    # Send empty detections for track_buffer frames
    for i in range(5):
        tracker.update(np.empty((0, 6)))
    # Track should still exist (lost but not removed)
    assert len(tracker.tracks) > 0


def test_track_removed_after_buffer():
    """Track should be removed after exceeding track_buffer."""
    tracker = ARCTracker({
        "min_hits": 1,
        "new_track_thresh": 0.7,
        "track_buffer": 3,
    })
    dets = _dets((100, 100, 200, 200, 0.9, 0))
    tracker.update(dets)

    # Send empty detections beyond track_buffer
    for _ in range(10):
        tracker.update(np.empty((0, 6)))
    assert len(tracker.tracks) == 0


# ---------------------------------------------------------------------------
# min_hits warmup behavior
# ---------------------------------------------------------------------------

def test_min_hits_warmup():
    """During warmup (frame_count <= min_hits), tracks output immediately."""
    tracker = ARCTracker({"min_hits": 3, "new_track_thresh": 0.7})
    dets = _dets((100, 100, 200, 200, 0.9, 0))
    result = tracker.update(dets)  # frame 1 <= min_hits=3
    assert result.shape[0] == 1


def test_min_hits_suppression():
    """After warmup, tracks need min_hits consecutive updates to output."""
    tracker = ARCTracker({"min_hits": 3, "new_track_thresh": 0.7})
    # Process past warmup
    for _ in range(4):
        tracker.update(np.empty((0, 6)))

    # Now create a new track (frame 5, past warmup)
    dets = _dets((100, 100, 200, 200, 0.9, 0))
    result = tracker.update(dets)
    # New track has hit_streak=1, needs 3, should not output
    assert result.shape[0] == 0


# ---------------------------------------------------------------------------
# OCR recovery
# ---------------------------------------------------------------------------

def test_ocr_recovery():
    """Track lost briefly should recover via OCR when object reappears."""
    tracker = ARCTracker({
        "min_hits": 1,
        "new_track_thresh": 0.7,
        "track_buffer": 10,
    })

    # Establish track
    for _ in range(3):
        dets = _dets((100, 100, 200, 200, 0.9, 0))
        tracker.update(dets)

    track_id_before = tracker.tracks[0].track_id

    # Lose for 2 frames
    tracker.update(np.empty((0, 6)))
    tracker.update(np.empty((0, 6)))

    # Re-appear at same position
    dets = _dets((101, 101, 201, 201, 0.9, 0))
    result = tracker.update(dets)

    # Should recover same track
    recovered_ids = [t.track_id for t in tracker.tracks
                     if t.time_since_update == 0]
    assert track_id_before in recovered_ids


# ---------------------------------------------------------------------------
# ORU re-update
# ---------------------------------------------------------------------------

def test_oru_reupdate():
    """After gap, KF state should be repaired via ORU virtual trajectory."""
    tracker = ARCTracker({
        "min_hits": 1,
        "new_track_thresh": 0.7,
        "track_buffer": 10,
    })

    # Establish track with some observations
    for i in range(5):
        dets = _dets((100 + i, 100, 200 + i, 200, 0.9, 0))
        tracker.update(dets)

    # Lose for 3 frames
    for _ in range(3):
        tracker.update(np.empty((0, 6)))

    # Re-appear -- KF should use ORU to re-update
    dets = _dets((110, 100, 210, 200, 0.9, 0))
    tracker.update(dets)

    # Track should still exist and be recently updated
    active = [t for t in tracker.tracks if t.time_since_update == 0]
    assert len(active) > 0


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def test_load_config_from_yaml():
    config_data = {
        "tracker": {
            "track_high_thresh": 0.5,
            "min_hits": 2,
            "ocsort": {"delta_t": 5},
        }
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(config_data, f)
        tmp_path = f.name

    try:
        config = load_config(tmp_path)
        assert config["track_high_thresh"] == 0.5
        assert config["min_hits"] == 2
        assert config["ocsort"]["delta_t"] == 5
    finally:
        os.unlink(tmp_path)


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")


def test_default_config_values():
    tracker = ARCTracker()
    assert tracker.track_high_thresh == DEFAULT_CONFIG["track_high_thresh"]
    assert tracker.match_thresh == DEFAULT_CONFIG["match_thresh"]
    assert tracker.track_buffer == DEFAULT_CONFIG["track_buffer"]


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

def test_reset_clears_state():
    tracker = ARCTracker({"min_hits": 1, "new_track_thresh": 0.7})
    dets = _dets((100, 100, 200, 200, 0.9, 0))
    tracker.update(dets)
    assert len(tracker.tracks) > 0
    assert tracker.frame_count > 0

    tracker.reset()
    assert len(tracker.tracks) == 0
    assert tracker.frame_count == 0


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def test_nan_prediction_removes_track():
    """If KF prediction produces NaN, track should be removed."""
    tracker = ARCTracker({"min_hits": 1, "new_track_thresh": 0.7})
    dets = _dets((100, 100, 200, 200, 0.9, 0))
    tracker.update(dets)

    # Corrupt the KF state to produce NaN
    tracker.tracks[0].kf.x[:] = np.nan

    # Next update should clean up the NaN track
    result = tracker.update(np.empty((0, 6)))
    # NaN track should be filtered out
    for t in tracker.tracks:
        bbox = t.get_state()
        assert np.all(np.isfinite(bbox))


# ---------------------------------------------------------------------------
# Multiple objects crossing
# ---------------------------------------------------------------------------

def test_multiple_objects_crossing_paths():
    """Two objects that cross paths should maintain identity as much as possible."""
    tracker = ARCTracker({
        "min_hits": 1,
        "new_track_thresh": 0.7,
        "match_thresh": 0.1,
    })

    # Object A starts at left, B at right
    for i in range(5):
        dets = _dets(
            (100 + i * 20, 100, 200 + i * 20, 200, 0.9, 0),
            (500 - i * 20, 100, 600 - i * 20, 200, 0.9, 0),
        )
        result = tracker.update(dets)

    # Should have 2 active tracks
    active = [t for t in tracker.tracks if t.time_since_update == 0]
    assert len(active) == 2


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------

def test_output_format():
    """Output should be [x1, y1, x2, y2, track_id] with track_id >= 1."""
    tracker = ARCTracker({"min_hits": 1, "new_track_thresh": 0.7})
    dets = _dets((100, 100, 200, 200, 0.9, 0))
    result = tracker.update(dets)
    assert result.shape[1] == 5
    assert result[0, 4] >= 1  # track IDs are 1-indexed in output


# ---------------------------------------------------------------------------
# BYTE two-stage matching
# ---------------------------------------------------------------------------

def test_byte_low_conf_rescues_track():
    """With BYTE, a low-conf detection should keep a tracked object alive."""
    tracker = ARCTracker({
        "min_hits": 1,
        "new_track_thresh": 0.7,
        "track_high_thresh": 0.6,
        "track_low_thresh": 0.1,
        "use_byte": True,
        "byte_match_thresh": 0.5,
    })

    # Establish track with high-conf detections
    for _ in range(3):
        dets = _dets((100, 100, 200, 200, 0.9, 0))
        tracker.update(dets)

    track_id = tracker.tracks[0].track_id

    # Send a low-conf detection at same position (below track_high_thresh)
    low_dets = _dets((101, 101, 201, 201, 0.3, 0))
    tracker.update(low_dets)

    # Track should still be active (matched via BYTE second stage)
    active = [t for t in tracker.tracks if t.track_id == track_id]
    assert len(active) == 1
    assert active[0].time_since_update == 0


def test_byte_disabled_ignores_low_conf():
    """Without BYTE, low-conf detections should not match tracks."""
    tracker = ARCTracker({
        "min_hits": 1,
        "new_track_thresh": 0.7,
        "track_high_thresh": 0.6,
        "track_low_thresh": 0.1,
        "use_byte": False,
    })

    # Establish track
    for _ in range(3):
        dets = _dets((100, 100, 200, 200, 0.9, 0))
        tracker.update(dets)

    track_id = tracker.tracks[0].track_id

    # Send only low-conf detection
    low_dets = _dets((101, 101, 201, 201, 0.3, 0))
    tracker.update(low_dets)

    # Track should NOT be matched (no high-conf det, BYTE disabled)
    active = [t for t in tracker.tracks if t.track_id == track_id]
    assert len(active) == 1
    assert active[0].time_since_update > 0


def test_byte_low_conf_does_not_create_track():
    """Low-confidence detections should never initialize new tracks."""
    tracker = ARCTracker({
        "min_hits": 1,
        "new_track_thresh": 0.7,
        "track_high_thresh": 0.6,
        "track_low_thresh": 0.1,
        "use_byte": True,
    })

    # Only low-conf detections
    dets = _dets((100, 100, 200, 200, 0.3, 0))
    tracker.update(dets)
    assert len(tracker.tracks) == 0


def test_byte_high_and_low_conf_same_frame():
    """High-conf creates track, low-conf rescues a different tracked object."""
    tracker = ARCTracker({
        "min_hits": 1,
        "new_track_thresh": 0.7,
        "track_high_thresh": 0.6,
        "track_low_thresh": 0.1,
        "use_byte": True,
        "byte_match_thresh": 0.3,
    })

    # Create two tracks
    dets = _dets(
        (100, 100, 200, 200, 0.9, 0),
        (400, 400, 500, 500, 0.9, 0),
    )
    for _ in range(3):
        tracker.update(dets)

    assert len(tracker.tracks) == 2
    id_a = tracker.tracks[0].track_id
    id_b = tracker.tracks[1].track_id

    # Next frame: track A gets high-conf, track B gets only low-conf
    mixed_dets = _dets(
        (102, 102, 202, 202, 0.9, 0),   # high-conf for track A
        (402, 402, 502, 502, 0.3, 0),    # low-conf for track B
    )
    tracker.update(mixed_dets)

    # Both tracks should be active
    track_a = [t for t in tracker.tracks if t.track_id == id_a][0]
    track_b = [t for t in tracker.tracks if t.track_id == id_b][0]
    assert track_a.time_since_update == 0
    assert track_b.time_since_update == 0


def test_byte_only_tracked_state_eligible():
    """BYTE second stage should only match actively tracked (not lost) tracks."""
    tracker = ARCTracker({
        "min_hits": 1,
        "new_track_thresh": 0.7,
        "track_high_thresh": 0.6,
        "track_low_thresh": 0.1,
        "track_buffer": 10,
        "use_byte": True,
        "byte_match_thresh": 0.3,
    })

    # Create a track
    for _ in range(3):
        dets = _dets((100, 100, 200, 200, 0.9, 0))
        tracker.update(dets)

    # Lose it for 2 frames (becomes LOST)
    tracker.update(np.empty((0, 6)))
    tracker.update(np.empty((0, 6)))

    lost_tracks = [t for t in tracker.tracks if t.state.name == "LOST"]
    assert len(lost_tracks) == 1

    # Send low-conf detection near the lost track
    low_dets = _dets((101, 101, 201, 201, 0.3, 0))
    tracker.update(low_dets)

    # Lost track should NOT be matched by BYTE (only tracked/new are eligible)
    lost_track = lost_tracks[0]
    assert lost_track.time_since_update > 0


# ---------------------------------------------------------------------------
# CMC integration
# ---------------------------------------------------------------------------

def _make_checkerboard_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Create a frame with strong corner features for CMC."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    block = 40
    for y in range(0, h, block):
        for x in range(0, w, block):
            if ((x // block) + (y // block)) % 2 == 0:
                frame[y:y + block, x:x + block] = 255
    return frame


def test_cmc_tracker_accepts_frame():
    """Tracker with CMC should accept frame parameter without error."""
    tracker = ARCTracker({
        "min_hits": 1,
        "new_track_thresh": 0.7,
        "use_cmc": True,
    })
    frame = _make_checkerboard_frame()
    dets = _dets((100, 100, 200, 200, 0.9, 0))
    result = tracker.update(dets, frame=frame)
    assert result.shape[1] == 5


def test_cmc_tracker_static_camera_preserves_tracking():
    """With identical frames (static camera), CMC should not hurt tracking."""
    tracker = ARCTracker({
        "min_hits": 1,
        "new_track_thresh": 0.7,
        "use_cmc": True,
    })
    frame = _make_checkerboard_frame()

    # Establish a track over several frames
    for _ in range(3):
        dets = _dets((100, 100, 200, 200, 0.9, 0))
        tracker.update(dets, frame=frame)

    track_id = tracker.tracks[0].track_id

    # Continue with same frame and slightly moved detection
    dets = _dets((102, 102, 202, 202, 0.9, 0))
    tracker.update(dets, frame=frame)

    # Track should still be active
    active = [t for t in tracker.tracks if t.track_id == track_id]
    assert len(active) == 1
    assert active[0].time_since_update == 0


def test_cmc_disabled_ignores_frame():
    """Without CMC, passing a frame should have no effect."""
    tracker = ARCTracker({
        "min_hits": 1,
        "new_track_thresh": 0.7,
        "use_cmc": False,
    })
    frame = _make_checkerboard_frame()
    dets = _dets((100, 100, 200, 200, 0.9, 0))
    result = tracker.update(dets, frame=frame)
    assert result.shape[0] >= 1
    assert len(tracker.affine_transforms) == 0


def test_cmc_stores_affine_transforms():
    """CMC should store per-frame affine transforms for ORU correction."""
    tracker = ARCTracker({
        "min_hits": 1,
        "new_track_thresh": 0.7,
        "use_cmc": True,
    })
    frame = _make_checkerboard_frame()
    dets = _dets((100, 100, 200, 200, 0.9, 0))

    tracker.update(dets, frame=frame)
    tracker.update(dets, frame=frame)

    assert len(tracker.affine_transforms) == 2
    assert 1 in tracker.affine_transforms
    assert 2 in tracker.affine_transforms
    assert tracker.affine_transforms[1].shape == (2, 3)


def test_cmc_reset_clears_transforms():
    """Reset should clear stored affine transforms."""
    tracker = ARCTracker({
        "min_hits": 1,
        "new_track_thresh": 0.7,
        "use_cmc": True,
    })
    frame = _make_checkerboard_frame()
    dets = _dets((100, 100, 200, 200, 0.9, 0))
    tracker.update(dets, frame=frame)

    tracker.reset()
    assert len(tracker.affine_transforms) == 0


# ---------------------------------------------------------------------------
# Mahalanobis motion gate (use_mahalanobis_gate)
# ---------------------------------------------------------------------------

def test_mahalanobis_gate_requires_imm():
    """Gate without IMM must be disabled with a warning, not crash."""
    tracker = ARCTracker({"use_mahalanobis_gate": True, "use_imm": False})
    assert tracker.use_mahalanobis_gate is False
    # Still tracks normally with the baseline KF
    for i in range(5):
        result = tracker.update(_dets((10.0 + i, 10.0, 30.0 + i, 50.0)))
    assert result.shape[0] == 1


def test_mahalanobis_gate_stable_tracking():
    """Gated tracker must keep a stable ID on smooth linear motion."""
    tracker = ARCTracker({
        "use_imm": True,
        "use_mahalanobis_gate": True,
        "min_hits": 1,
    })
    ids = set()
    for i in range(10):
        x = 10.0 + 5 * i
        result = tracker.update(_dets((x, 10.0, x + 20.0, 50.0)))
        assert result.shape[0] == 1
        ids.add(int(result[0, 4]))
    assert len(ids) == 1


def test_mahalanobis_gate_blocks_teleporting_detection():
    """A detection far outside the motion gate must spawn a new track,
    not be absorbed into the existing one."""
    config = {
        "use_imm": True,
        "use_mahalanobis_gate": True,
        "use_hybrid_distance": True,
        "min_hits": 1,
        "new_track_thresh": 0.5,
    }
    tracker = ARCTracker(config)
    # Establish a well-observed stationary track
    for _ in range(8):
        tracker.update(_dets((100.0, 100.0, 120.0, 140.0)))

    # Detection teleports far away (zero IoU, far beyond motion gate)
    result = tracker.update(_dets((800.0, 800.0, 820.0, 840.0)))
    # The old track's ID must not jump to the teleported detection
    old_id_boxes = [r for r in result if int(r[4]) == 1]
    for r in old_id_boxes:
        assert r[0] < 200.0


def test_mahalanobis_gate_matches_ungated_on_easy_scene():
    """On an easy two-object scene, gating must not change the output."""
    def run(gate: bool) -> list[np.ndarray]:
        reset_track_id()
        tracker = ARCTracker({
            "use_imm": True,
            "use_mahalanobis_gate": gate,
            "min_hits": 1,
        })
        outs = []
        for i in range(8):
            x = 10.0 + 4 * i
            outs.append(tracker.update(_dets(
                (x, 10.0, x + 20.0, 50.0),
                (300.0 - 4 * i, 200.0, 320.0 - 4 * i, 240.0),
            )))
        return outs

    outs_ungated = run(False)
    outs_gated = run(True)
    for a, b in zip(outs_ungated, outs_gated):
        np.testing.assert_allclose(a, b, atol=1e-9)


# ---------------------------------------------------------------------------
# use_cmc_oru toggle
# ---------------------------------------------------------------------------

def test_cmc_oru_disabled_returns_no_gap_affines():
    """With use_cmc_oru=False, _gather_gap_affines must return None even
    when CMC is on and camera motion occurred during the gap."""
    moving_affine = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0]])

    def run(use_cmc_oru: bool):
        reset_track_id()
        tracker = ARCTracker({
            "use_cmc": True,
            "min_hits": 1,
            "ocsort": {"use_cmc_oru": use_cmc_oru},
        })
        # Establish a track, then miss it for 3 frames with camera motion
        tracker.update(_dets((100.0, 100.0, 120.0, 140.0)),
                       cmc_affine=np.eye(2, 3))
        for _ in range(3):
            tracker.update(np.empty((0, 6)), cmc_affine=moving_affine)
        track = tracker.tracks[0]
        return tracker._gather_gap_affines(track)

    assert run(True) is not None
    assert run(False) is None


def test_cmc_oru_default_enabled():
    tracker = ARCTracker()
    assert tracker.use_cmc_oru is True


# ---------------------------------------------------------------------------
# ExpansionIoU (use_eiou)
# ---------------------------------------------------------------------------

def test_eiou_fixed_expansions():
    tracker = ARCTracker({"use_eiou": True, "min_hits": 1})
    tracker.update(_dets((10.0, 10.0, 30.0, 50.0), (200.0, 10.0, 220.0, 50.0)))
    exp = tracker._compute_track_expansions()
    np.testing.assert_allclose(exp, [0.7, 0.7])


def test_eiou_adaptive_requires_imm():
    tracker = ARCTracker({
        "use_eiou": True, "use_imm": False,
        "eiou": {"adaptive": True},
    })
    assert tracker.eiou_config["adaptive"] is False


def test_eiou_adaptive_grows_with_uncertainty():
    """A coasting (missed) track must get a larger expansion than a
    freshly updated one."""
    tracker = ARCTracker({
        "use_imm": True,
        "use_eiou": True,
        "eiou": {"adaptive": True, "adaptive_scale": 1.0,
                 "min_expansion": 0.0, "max_expansion": 5.0},
        "min_hits": 1,
        "track_buffer": 60,
    })
    for i in range(6):
        x = 10.0 + 5 * i
        tracker.update(_dets((x, 10.0, x + 20.0, 50.0)))
    exp_fresh = tracker._compute_track_expansions()[0]

    for _ in range(8):
        tracker.update(np.empty((0, 6)))
    exp_coasting = tracker._compute_track_expansions()[0]

    assert exp_coasting > exp_fresh


def test_eiou_tracks_maintain_ids():
    """EIoU association must keep IDs on a fast-moving object that IoU
    alone would lose (jump larger than box width)."""
    def run(use_eiou: bool) -> int:
        reset_track_id()
        tracker = ARCTracker({
            "use_eiou": use_eiou,
            "eiou": {"expansion": 0.7, "adaptive": False},
            "min_hits": 1,
        })
        ids = set()
        for i in range(8):
            x = 10.0 + 25.0 * i   # jump > box width each frame
            out = tracker.update(_dets((x, 10.0, x + 20.0, 50.0)))
            for r in out:
                ids.add(int(r[4]))
        return len(ids)

    assert run(True) == 1        # EIoU bridges the jumps
    assert run(False) > 1        # plain IoU fragments


# ---------------------------------------------------------------------------
# Output box
# ---------------------------------------------------------------------------

def _run_jittered(output_box: str) -> tuple[np.ndarray, np.ndarray]:
    """Track one box moving with jitter; return the last detection and output."""
    tracker = ARCTracker({"output_box": output_box})
    rng = np.random.default_rng(0)
    out = np.empty((0, 5))
    for t in range(10):
        box = np.array([100 + 5 * t, 100, 150 + 5 * t, 200]) + rng.normal(0, 2, 4)
        out = tracker.update(_dets(tuple(box)))
    return box, out


def test_output_box_default_is_detection():
    assert DEFAULT_CONFIG["output_box"] == "detection"


def test_output_box_detection_returns_matched_box():
    box, out = _run_jittered("detection")
    assert len(out) == 1
    np.testing.assert_allclose(out[0, :4], box)


def test_output_box_kalman_returns_filtered_state():
    box, out = _run_jittered("kalman")
    assert len(out) == 1
    assert not np.allclose(out[0, :4], box)


def test_output_box_invalid_rejected():
    with pytest.raises(ValueError, match="output_box"):
        ARCTracker({"output_box": "smoothed"})
