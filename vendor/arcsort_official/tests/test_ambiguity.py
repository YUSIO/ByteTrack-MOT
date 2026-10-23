"""Tests for the ambiguity instrument (swap margins, association diagnostics)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from arcsort.association import associate, swap_margins
from arcsort.tracker import ARCTracker


# ---------------------------------------------------------------------------
# swap_margins
# ---------------------------------------------------------------------------

def test_swap_margin_small_for_near_degenerate_pair():
    # matches (0,0) and (1,1); permutation nearly as good
    score = np.array([
        [0.80, 0.75],
        [0.74, 0.81],
    ])
    spatial = score.copy()
    matched = np.array([[0, 0], [1, 1]])
    out = swap_margins(matched, score, spatial)
    assert len(out) == 1
    a, b, margin = out[0]
    assert {a, b} == {0, 1}
    assert abs(margin - (0.80 + 0.81 - 0.75 - 0.74)) < 1e-9  # 0.12


def test_swap_margin_requires_spatial_support_both_ways():
    # cross-assignments have zero spatial overlap -> not comparable,
    # even though OCM-style bonuses could make total score positive
    score = np.array([
        [0.90, 0.10],
        [0.10, 0.90],
    ])
    spatial = np.array([
        [0.90, 0.00],
        [0.00, 0.90],
    ])
    matched = np.array([[0, 0], [1, 1]])
    assert swap_margins(matched, score, spatial) == []


def test_swap_margin_large_for_clear_assignment():
    score = np.array([
        [0.95, 0.20],
        [0.15, 0.90],
    ])
    matched = np.array([[0, 0], [1, 1]])
    out = swap_margins(matched, score, score)
    assert len(out) == 1
    assert out[0][2] > 1.0  # 0.95+0.90-0.20-0.15 = 1.50


def test_swap_margin_needs_two_matches():
    assert swap_margins(np.array([[0, 0]]), np.eye(1), np.eye(1)) == []
    assert swap_margins(np.empty((0, 2), dtype=int), np.eye(1), np.eye(1)) == []


# ---------------------------------------------------------------------------
# associate diagnostics
# ---------------------------------------------------------------------------

def _two_crossing_targets():
    dets = np.array([
        [100.0, 100.0, 140.0, 180.0],
        [110.0, 100.0, 150.0, 180.0],
    ])
    preds = np.array([
        [102.0, 100.0, 142.0, 180.0],
        [108.0, 100.0, 148.0, 180.0],
    ])
    vels = np.zeros((2, 2))
    prev_obs = np.hstack([preds, np.ones((2, 1))])
    return dets, preds, vels, prev_obs


def test_associate_diagnostics_do_not_change_assignment():
    dets, preds, vels, prev_obs = _two_crossing_targets()
    plain = associate(dets, preds, vels, prev_obs,
                      iou_threshold=0.25, inertia_weight=0.2)
    with_diag = associate(dets, preds, vels, prev_obs,
                          iou_threshold=0.25, inertia_weight=0.2,
                          return_diagnostics=True)
    assert len(with_diag) == 4
    for i in range(3):
        np.testing.assert_array_equal(plain[i], with_diag[i])
    diag = with_diag[3]
    assert diag["score"].shape == (2, 2)
    assert diag["spatial"].shape == (2, 2)


def test_crossing_pair_yields_low_margin_event():
    dets, preds, vels, prev_obs = _two_crossing_targets()
    matched, _, _, diag = associate(dets, preds, vels, prev_obs,
                                    iou_threshold=0.25, inertia_weight=0.2,
                                    return_diagnostics=True)
    assert matched.shape[0] == 2
    out = swap_margins(matched, diag["score"], diag["spatial"])
    assert len(out) == 1
    assert out[0][2] < 0.5  # heavily overlapping pair: near-ambiguous


# ---------------------------------------------------------------------------
# tracker wiring
# ---------------------------------------------------------------------------

def test_tracker_default_has_instrument_off():
    tracker = ARCTracker({"min_hits": 1})
    assert tracker.ambiguity_enabled is False
    assert tracker.pop_ambiguity_events() == []


def test_tracker_records_events_when_enabled():
    tracker = ARCTracker({
        "min_hits": 1,
        "new_track_thresh": 0.5,
        "ambiguity": {"enabled": True, "margin_thresh": 0.3},
    })
    # two targets converging to near-coincidence (a crossing)
    for f in range(8):
        x1 = 100.0 + 5 * f
        x2 = 170.0 - 5 * f
        dets = np.array([
            [x1, 100.0, x1 + 40.0, 180.0, 0.9, 0.0],
            [x2, 100.0, x2 + 40.0, 180.0, 0.9, 0.0],
        ])
        tracker.update(dets)
    events = tracker.pop_ambiguity_events()
    assert len(events) > 0
    ev = events[0]
    assert {"frame", "track_a", "track_b", "margin"} <= set(ev)
    assert ev["track_a"] != ev["track_b"]


def test_tracker_reset_clears_events():
    tracker = ARCTracker({
        "min_hits": 1,
        "new_track_thresh": 0.5,
        "ambiguity": {"enabled": True, "margin_thresh": 0.5},
    })
    for f in range(8):
        dets = np.array([
            [100.0, 100.0, 140.0, 180.0, 0.9, 0.0],
            [120.0, 100.0, 160.0, 180.0, 0.9, 0.0],
        ])
        tracker.update(dets)
    tracker.reset()
    assert tracker.pop_ambiguity_events() == []
