"""Tests for deferred identity arbitration (corrective ID swap)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from arcsort.tracker import ARCTracker


def _unit(dim_idx, dim=512):
    v = np.zeros(dim, dtype=np.float32)
    v[dim_idx] = 1.0
    return v


# Costume-degenerate identities (the DanceTrack regime): a dominant
# shared component plus a small identity component, cos(A, B) ≈ 0.95.
# Appearance then cannot decide the crossing (margin below threshold),
# the spatial term drives a wrong straight-through call, and only
# evidence integrated over many post-cross frames can flip it.
_COMMON, _ID_A, _ID_B = _unit(0), _unit(1), _unit(2)
_W = 0.975
EMB_A = (_W * _COMMON + np.sqrt(1 - _W**2) * _ID_A).astype(np.float32)
EMB_B = (_W * _COMMON + np.sqrt(1 - _W**2) * _ID_B).astype(np.float32)


def _cache_npz(tmp_path, frames):
    arrays = {}
    for fid, (boxes, embs) in frames.items():
        arrays[f"boxes_{fid}"] = boxes.astype(np.float32)
        arrays[f"embs_{fid}"] = embs.astype(np.float16)
    path = tmp_path / "seq.npz"
    np.savez_compressed(path, **arrays)
    return path


def _crossing_scenario(swap_embeddings_after_cross: bool):
    """Two targets cross at frame ~8; if swap_embeddings_after_cross,
    the embeddings travel THROUGH the crossing with the boxes (i.e. the
    spatial tracker will keep straight-line identities, which is wrong —
    the people bounced off each other), so arbitration must swap IDs.

    Frames 1..20. Target boxes converge, coincide at frame 8, separate.
    """
    frames = {}
    for fid in range(1, 21):
        t = fid - 1
        x1 = 100.0 + 10.0 * t          # left target moving right
        x2 = 240.0 - 10.0 * t          # right target moving left
        boxes = np.array([
            [x1, 100.0, x1 + 40.0, 180.0],
            [x2, 100.0, x2 + 40.0, 180.0],
        ])
        if swap_embeddings_after_cross and x1 > x2:
            embs = np.stack([EMB_B, EMB_A])   # appearances crossed over
        else:
            embs = np.stack([EMB_A, EMB_B])
        frames[fid] = (boxes, embs)
    return frames


def _run(tmp_path, frames, arbitration: bool):
    cfg = {
        "use_reid": True,
        "min_hits": 1,
        "new_track_thresh": 0.5,
        "arbitration": {
            "enabled": arbitration,
            "window": 8,
            "min_frames": 3,
            "llr_thresh": 0.5,
        },
        "ambiguity": {"enabled": False, "margin_thresh": 0.2},
    }
    tracker = ARCTracker(cfg)
    tracker.embedding_computer.load_cache(_cache_npz(tmp_path, frames))
    id_of_left = []
    for fid in sorted(frames):
        boxes, _ = frames[fid]
        dets = np.hstack([boxes, np.full((2, 1), 0.9), np.zeros((2, 1))])
        out = tracker.update(dets, frame=None)
        if out.shape[0] == 2:
            left_row = out[np.argmin(out[:, 0])]
            id_of_left.append(int(left_row[4]))
    return tracker, id_of_left


def test_wrong_crossing_decision_is_corrected_by_swap(tmp_path):
    frames = _crossing_scenario(swap_embeddings_after_cross=True)
    tracker, id_of_left = _run(tmp_path, frames, arbitration=True)
    assert len(tracker.arbitration_swaps) == 1
    # the people bounced: person A starts AND ends on the left. The
    # spatial tracker goes straight through (left side would end with
    # the other id); the corrective swap restores the original id.
    assert id_of_left[0] == id_of_left[-1]

    _, id_of_left_off = _run(tmp_path, frames, arbitration=False)
    assert id_of_left_off[0] != id_of_left_off[-1]  # uncorrected


def test_correct_crossing_decision_is_left_alone(tmp_path):
    frames = _crossing_scenario(swap_embeddings_after_cross=False)
    tracker, id_of_left = _run(tmp_path, frames, arbitration=True)
    assert tracker.arbitration_swaps == []


def test_arbitration_off_never_swaps(tmp_path):
    frames = _crossing_scenario(swap_embeddings_after_cross=True)
    tracker, _ = _run(tmp_path, frames, arbitration=False)
    assert tracker.arbitration_swaps == []
    assert tracker.arb_episodes == {}


def test_at_most_one_swap_per_episode(tmp_path):
    frames = _crossing_scenario(swap_embeddings_after_cross=True)
    tracker, _ = _run(tmp_path, frames, arbitration=True)
    assert len(tracker.arbitration_swaps) <= 1
