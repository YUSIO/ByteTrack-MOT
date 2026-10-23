"""Tests for tools/merge_tracklets.py -- offline tracklet association."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.merge_tracklets import (
    Tracklet,
    attach_embeddings,
    load_tracks,
    merge_tracklets,
    process_sequence,
    split_tracklets,
)


def _unit(active_dims, dim=64):
    v = np.zeros(dim, dtype=np.float32)
    for d, val in active_dims:
        v[d] = val
    return v / np.linalg.norm(v)


def _tracklet(tid, frames, x_start, vx=5.0, emb=None, y=100.0, size=40.0):
    t = Tracklet(tid)
    for i, f in enumerate(frames):
        x = x_start + vx * i
        t.frames.append(f)
        t.boxes.append(np.array([x, y, x + size, y + 2 * size]))
        if emb is not None:
            t.embs.append(emb)
    return t


# ---------------------------------------------------------------------------
# merge_tracklets
# ---------------------------------------------------------------------------

def test_merge_same_identity_across_gap():
    """Disjoint tracklets, same appearance, motion-consistent -> merged."""
    emb = _unit([(0, 1.0)])
    a = _tracklet(1, range(1, 21), x_start=0.0, emb=emb)
    # B starts where A's extrapolation lands after a 10-frame gap
    b = _tracklet(2, range(31, 51), x_start=5.0 * 30, emb=emb)
    mapping = merge_tracklets({1: a, 2: b})
    assert mapping[1] == mapping[2]


def test_no_merge_different_identity():
    """Orthogonal appearance must block the merge despite good motion."""
    a = _tracklet(1, range(1, 21), 0.0, emb=_unit([(0, 1.0)]))
    b = _tracklet(2, range(31, 51), 150.0, emb=_unit([(1, 1.0)]))
    mapping = merge_tracklets({1: a, 2: b})
    assert mapping[1] != mapping[2]


def test_no_merge_motion_incompatible():
    """Same appearance but teleported far from the extrapolation -> no merge."""
    emb = _unit([(0, 1.0)])
    a = _tracklet(1, range(1, 21), 0.0, emb=emb)
    b = _tracklet(2, range(31, 51), 5000.0, emb=emb)
    mapping = merge_tracklets({1: a, 2: b})
    assert mapping[1] != mapping[2]


def test_no_merge_temporal_overlap():
    """Temporally overlapping tracklets are never merge candidates."""
    emb = _unit([(0, 1.0)])
    a = _tracklet(1, range(1, 30), 0.0, emb=emb)
    b = _tracklet(2, range(25, 50), 120.0, emb=emb)
    mapping = merge_tracklets({1: a, 2: b})
    assert mapping[1] != mapping[2]


def test_merge_chain_resolves_to_single_root():
    """A -> B -> C chains must all map to one root id."""
    emb = _unit([(0, 1.0)])
    a = _tracklet(1, range(1, 11), 0.0, emb=emb)
    b = _tracklet(2, range(16, 26), 5.0 * 15, emb=emb)
    c = _tracklet(3, range(31, 41), 5.0 * 30, emb=emb)
    mapping = merge_tracklets({1: a, 2: b, 3: c})
    assert mapping[1] == mapping[2] == mapping[3]


def test_merge_tail_used_once():
    """One tail may claim only its best head, not several."""
    emb = _unit([(0, 1.0)])
    a = _tracklet(1, range(1, 11), 0.0, emb=emb)
    b1 = _tracklet(2, range(16, 26), 5.0 * 15, emb=emb)
    b2 = _tracklet(3, range(18, 28), 5.0 * 17 + 3, emb=emb)
    mapping = merge_tracklets({1: a, 2: b1, 3: b2})
    merged_with_a = [t for t in (2, 3) if mapping[t] == mapping[1]]
    assert len(merged_with_a) == 1


# ---------------------------------------------------------------------------
# split_tracklets
# ---------------------------------------------------------------------------

def _cache_for_track(track, embs_by_frame, dim=64):
    """Build an embedding cache aligned with a track's boxes."""
    cache = {}
    for frame, box in zip(track.frames, track.boxes):
        emb = embs_by_frame(frame)
        cache[frame] = (
            box[np.newaxis, :].astype(np.float64),
            emb[np.newaxis, :].astype(np.float32),
        )
    return cache


def test_split_at_identity_flip():
    """Track whose appearance flips halfway must split into two ids."""
    emb_a, emb_b = _unit([(0, 1.0)]), _unit([(1, 1.0)])
    track = _tracklet(1, range(1, 41), 0.0)
    cache = _cache_for_track(
        track, lambda f: emb_a if f <= 20 else emb_b
    )
    out = split_tracklets({1: track}, cache, split_thresh=0.45, persist=5)
    assert len(out) == 2
    lengths = sorted(len(t.frames) for t in out.values())
    assert lengths[0] >= 15  # split near the flip, not at the ends


def test_no_split_consistent_identity():
    emb = _unit([(0, 1.0)])
    track = _tracklet(1, range(1, 41), 0.0)
    cache = _cache_for_track(track, lambda f: emb)
    out = split_tracklets({1: track}, cache)
    assert len(out) == 1


def test_no_split_on_transient_noise():
    """A brief appearance glitch (shorter than persist) must not split."""
    emb_a, emb_b = _unit([(0, 1.0)]), _unit([(1, 1.0)])
    track = _tracklet(1, range(1, 41), 0.0)
    cache = _cache_for_track(
        track, lambda f: emb_b if f in (20, 21) else emb_a
    )
    out = split_tracklets({1: track}, cache, persist=5)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# attach_embeddings / end-to-end
# ---------------------------------------------------------------------------

def test_attach_embeddings_iou_matching():
    """KF-shifted output boxes must still find their detection embedding."""
    emb = _unit([(0, 1.0)])
    track = _tracklet(1, [1, 2], 100.0)
    cache = {}
    for frame, box in zip(track.frames, track.boxes):
        shifted = box + np.array([3.0, 2.0, 3.0, 2.0])  # a few px off
        cache[frame] = (
            shifted[np.newaxis, :].astype(np.float64),
            emb[np.newaxis, :].astype(np.float32),
        )
    attach_embeddings({1: track}, cache)
    assert len(track.embs) == 2


def test_process_sequence_end_to_end(tmp_path):
    """Full pipeline: split-flip track + mergeable pair -> correct id count."""
    lines = []
    emb_a, emb_b = _unit([(0, 1.0)]), _unit([(1, 1.0)])
    cache_arrays = {}

    # Track 1: frames 1-40, identity flips at frame 21 (should split)
    for f in range(1, 41):
        x = 10.0 + 5 * f
        lines.append(f"{f},1,{x:.2f},100.00,40.00,80.00,1,-1,-1,-1")
        emb = emb_a if f <= 20 else emb_b
        cache_arrays.setdefault(f, ([], []))
        cache_arrays[f][0].append([x, 100.0, x + 40.0, 180.0])
        cache_arrays[f][1].append(emb)

    # Tracks 2+3: same identity, disjoint, motion-consistent (should merge)
    for f in range(1, 16):
        x = 500.0 + 4 * f
        lines.append(f"{f},2,{x:.2f},300.00,40.00,80.00,1,-1,-1,-1")
        cache_arrays.setdefault(f, ([], []))
        cache_arrays[f][0].append([x, 300.0, x + 40.0, 380.0])
        cache_arrays[f][1].append(_unit([(2, 1.0)]))
    for f in range(26, 41):
        x = 500.0 + 4 * f
        lines.append(f"{f},3,{x:.2f},300.00,40.00,80.00,1,-1,-1,-1")
        cache_arrays.setdefault(f, ([], []))
        cache_arrays[f][0].append([x, 300.0, x + 40.0, 380.0])
        cache_arrays[f][1].append(_unit([(2, 1.0)]))

    txt = tmp_path / "seq.txt"
    txt.write_text("\n".join(lines) + "\n")
    npz_arrays = {}
    for f, (boxes, embs) in cache_arrays.items():
        npz_arrays[f"boxes_{f}"] = np.asarray(boxes, dtype=np.float32)
        npz_arrays[f"embs_{f}"] = np.asarray(embs, dtype=np.float16)
    npz = tmp_path / "seq.npz"
    np.savez_compressed(npz, **npz_arrays)

    args = argparse.Namespace(
        split=True, split_thresh=0.45, split_persist=5,
        merge_thresh=0.6, max_gap=150, motion_tol=2.0,
    )
    out = tmp_path / "out" / "seq.txt"
    n_in, n_split, n_out = process_sequence(txt, npz, out, args)
    # 3 in -> split makes 4 -> merge of the pair makes 3
    assert n_in == 3
    assert n_split == 4
    assert n_out == 3

    result = load_tracks(out)
    assert len(result) == 3
