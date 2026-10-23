"""Tests for arcsort/association.py -- associate and associate_ocr."""

from __future__ import annotations

import numpy as np
import pytest

from arcsort.association import associate, associate_byte_second, associate_ocr


def _make_velocities(n: int) -> np.ndarray:
    """Create zero-velocity array for n tracks."""
    return np.zeros((n, 2))


def _make_prev_obs(bboxes: np.ndarray, conf: float = 0.9) -> np.ndarray:
    """Create previous obs array [x1,y1,x2,y2,conf] from bbox array."""
    n = bboxes.shape[0]
    confs = np.full((n, 1), conf)
    return np.hstack([bboxes, confs])


# ---------------------------------------------------------------------------
# associate -- basic matching
# ---------------------------------------------------------------------------

def test_associate_obvious_match():
    dets = np.array([[0.0, 0.0, 10.0, 10.0]])
    preds = np.array([[1.0, 1.0, 11.0, 11.0]])  # high IoU with det
    vels = _make_velocities(1)
    prev = _make_prev_obs(preds)
    matched, unmatched_d, unmatched_t = associate(
        dets, preds, vels, prev,
        iou_threshold=0.3, inertia_weight=0.0,
    )
    assert matched.shape[0] == 1
    assert matched[0, 0] == 0  # det idx
    assert matched[0, 1] == 0  # trk idx
    assert len(unmatched_d) == 0
    assert len(unmatched_t) == 0


def test_associate_no_overlap():
    dets = np.array([[0.0, 0.0, 10.0, 10.0]])
    preds = np.array([[100.0, 100.0, 110.0, 110.0]])
    vels = _make_velocities(1)
    prev = _make_prev_obs(preds)
    matched, unmatched_d, unmatched_t = associate(
        dets, preds, vels, prev,
        iou_threshold=0.3, inertia_weight=0.0,
    )
    assert matched.shape[0] == 0
    assert 0 in unmatched_d
    assert 0 in unmatched_t


def test_associate_empty_detections():
    dets = np.empty((0, 4))
    preds = np.array([[0.0, 0.0, 10.0, 10.0]])
    vels = _make_velocities(1)
    prev = _make_prev_obs(preds)
    matched, unmatched_d, unmatched_t = associate(
        dets, preds, vels, prev,
        iou_threshold=0.3, inertia_weight=0.0,
    )
    assert matched.shape[0] == 0
    assert len(unmatched_d) == 0
    assert 0 in unmatched_t


def test_associate_empty_tracks():
    dets = np.array([[0.0, 0.0, 10.0, 10.0]])
    preds = np.empty((0, 4))
    vels = np.empty((0, 2))
    prev = np.empty((0, 5))
    matched, unmatched_d, unmatched_t = associate(
        dets, preds, vels, prev,
        iou_threshold=0.3, inertia_weight=0.0,
    )
    assert matched.shape[0] == 0
    assert 0 in unmatched_d
    assert len(unmatched_t) == 0


def test_associate_multiple_matches():
    dets = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [50.0, 50.0, 60.0, 60.0],
    ])
    preds = np.array([
        [1.0, 1.0, 11.0, 11.0],
        [51.0, 51.0, 61.0, 61.0],
    ])
    vels = _make_velocities(2)
    prev = _make_prev_obs(preds)
    matched, unmatched_d, unmatched_t = associate(
        dets, preds, vels, prev,
        iou_threshold=0.3, inertia_weight=0.0,
    )
    assert matched.shape[0] == 2
    assert len(unmatched_d) == 0
    assert len(unmatched_t) == 0


def test_associate_iou_threshold_filtering():
    """Match with IoU below threshold should be rejected."""
    dets = np.array([[0.0, 0.0, 10.0, 10.0]])
    preds = np.array([[8.0, 8.0, 18.0, 18.0]])  # very low IoU
    vels = _make_velocities(1)
    prev = _make_prev_obs(preds)
    matched, unmatched_d, unmatched_t = associate(
        dets, preds, vels, prev,
        iou_threshold=0.5, inertia_weight=0.0,
    )
    # IoU of these boxes is ~4/196 ~ 0.02, well below 0.5
    assert matched.shape[0] == 0


def test_associate_more_dets_than_tracks():
    dets = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [50.0, 50.0, 60.0, 60.0],
        [100.0, 100.0, 110.0, 110.0],
    ])
    preds = np.array([[1.0, 1.0, 11.0, 11.0]])
    vels = _make_velocities(1)
    prev = _make_prev_obs(preds)
    matched, unmatched_d, unmatched_t = associate(
        dets, preds, vels, prev,
        iou_threshold=0.3, inertia_weight=0.0,
    )
    assert matched.shape[0] == 1
    assert len(unmatched_d) == 2
    assert len(unmatched_t) == 0


# ---------------------------------------------------------------------------
# associate_ocr -- recovery matching
# ---------------------------------------------------------------------------

def test_associate_ocr_recovery_match():
    dets = np.array([[0.0, 0.0, 10.0, 10.0]])
    last_obs = np.array([[1.0, 1.0, 11.0, 11.0]])  # high IoU
    matched, unmatched_d, unmatched_t = associate_ocr(
        dets, last_obs, iou_threshold=0.3,
    )
    assert matched.shape[0] == 1


def test_associate_ocr_no_match():
    dets = np.array([[0.0, 0.0, 10.0, 10.0]])
    last_obs = np.array([[100.0, 100.0, 110.0, 110.0]])
    matched, unmatched_d, unmatched_t = associate_ocr(
        dets, last_obs, iou_threshold=0.3,
    )
    assert matched.shape[0] == 0
    assert 0 in unmatched_d
    assert 0 in unmatched_t


def test_associate_ocr_empty_detections():
    dets = np.empty((0, 4))
    last_obs = np.array([[0.0, 0.0, 10.0, 10.0]])
    matched, unmatched_d, unmatched_t = associate_ocr(
        dets, last_obs, iou_threshold=0.3,
    )
    assert matched.shape[0] == 0
    assert len(unmatched_d) == 0
    assert 0 in unmatched_t


def test_associate_ocr_empty_tracks():
    dets = np.array([[0.0, 0.0, 10.0, 10.0]])
    last_obs = np.empty((0, 4))
    matched, unmatched_d, unmatched_t = associate_ocr(
        dets, last_obs, iou_threshold=0.3,
    )
    assert matched.shape[0] == 0
    assert 0 in unmatched_d
    assert len(unmatched_t) == 0


# ---------------------------------------------------------------------------
# associate_byte_second -- BYTE second-stage matching
# ---------------------------------------------------------------------------

def test_byte_second_obvious_match():
    """Low-conf detection overlapping a track prediction should match."""
    dets_low = np.array([[0.0, 0.0, 10.0, 10.0]])
    preds = np.array([[1.0, 1.0, 11.0, 11.0]])
    matched, unmatched_d, unmatched_t = associate_byte_second(
        dets_low, preds, iou_threshold=0.5,
    )
    assert matched.shape[0] == 1
    assert matched[0, 0] == 0
    assert matched[0, 1] == 0
    assert len(unmatched_d) == 0
    assert len(unmatched_t) == 0


def test_byte_second_no_overlap():
    """Non-overlapping low-conf det and track should not match."""
    dets_low = np.array([[0.0, 0.0, 10.0, 10.0]])
    preds = np.array([[100.0, 100.0, 110.0, 110.0]])
    matched, unmatched_d, unmatched_t = associate_byte_second(
        dets_low, preds, iou_threshold=0.5,
    )
    assert matched.shape[0] == 0
    assert 0 in unmatched_d
    assert 0 in unmatched_t


def test_byte_second_empty_dets():
    """Empty low-conf detections should return all tracks unmatched."""
    dets_low = np.empty((0, 4))
    preds = np.array([[0.0, 0.0, 10.0, 10.0]])
    matched, unmatched_d, unmatched_t = associate_byte_second(
        dets_low, preds, iou_threshold=0.5,
    )
    assert matched.shape[0] == 0
    assert len(unmatched_d) == 0
    assert 0 in unmatched_t


def test_byte_second_empty_tracks():
    """Empty track predictions should return all dets unmatched."""
    dets_low = np.array([[0.0, 0.0, 10.0, 10.0]])
    preds = np.empty((0, 4))
    matched, unmatched_d, unmatched_t = associate_byte_second(
        dets_low, preds, iou_threshold=0.5,
    )
    assert matched.shape[0] == 0
    assert 0 in unmatched_d
    assert len(unmatched_t) == 0


def test_byte_second_threshold_filtering():
    """Match with IoU below byte threshold should be rejected."""
    dets_low = np.array([[0.0, 0.0, 10.0, 10.0]])
    preds = np.array([[8.0, 8.0, 18.0, 18.0]])  # very low IoU ~0.02
    matched, unmatched_d, unmatched_t = associate_byte_second(
        dets_low, preds, iou_threshold=0.5,
    )
    assert matched.shape[0] == 0


def test_byte_second_multiple_matches():
    """Multiple low-conf dets matching multiple tracks."""
    dets_low = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [50.0, 50.0, 60.0, 60.0],
    ])
    preds = np.array([
        [1.0, 1.0, 11.0, 11.0],
        [51.0, 51.0, 61.0, 61.0],
    ])
    matched, unmatched_d, unmatched_t = associate_byte_second(
        dets_low, preds, iou_threshold=0.3,
    )
    assert matched.shape[0] == 2
    assert len(unmatched_d) == 0
    assert len(unmatched_t) == 0


# ---------------------------------------------------------------------------
# associate with hybrid distance
# ---------------------------------------------------------------------------

def test_associate_hybrid_rescues_zero_iou():
    """Hybrid distance should match nearby boxes with zero IoU."""
    # 1px gap -- IoU is zero but boxes are adjacent
    dets = np.array([[11.0, 0.0, 21.0, 10.0]])
    preds = np.array([[0.0, 0.0, 10.0, 10.0]])
    vels = _make_velocities(1)
    prev = _make_prev_obs(preds)

    # Without hybrid: should NOT match (IoU = 0)
    matched_plain, _, _ = associate(
        dets, preds, vels, prev,
        iou_threshold=0.01, inertia_weight=0.0,
        use_hybrid_distance=False,
    )
    assert matched_plain.shape[0] == 0

    # With hybrid: should match (boxes are close, low threshold)
    matched_hybrid, _, _ = associate(
        dets, preds, vels, prev,
        iou_threshold=0.01, inertia_weight=0.0,
        use_hybrid_distance=True,
    )
    assert matched_hybrid.shape[0] == 1


def test_associate_hybrid_still_rejects_distant():
    """Hybrid distance should still reject very distant boxes."""
    dets = np.array([[500.0, 500.0, 510.0, 510.0]])
    preds = np.array([[0.0, 0.0, 10.0, 10.0]])
    vels = _make_velocities(1)
    prev = _make_prev_obs(preds)

    matched, _, _ = associate(
        dets, preds, vels, prev,
        iou_threshold=0.1, inertia_weight=0.0,
        use_hybrid_distance=True,
    )
    assert matched.shape[0] == 0


# ---------------------------------------------------------------------------
# gate_mask -- Mahalanobis motion gating in first-stage association
# ---------------------------------------------------------------------------

def test_associate_gate_mask_blocks_match():
    """A pair forbidden by the gate must not match even with perfect IoU."""
    dets = np.array([[0.0, 0.0, 10.0, 10.0]])
    trks = np.array([[0.0, 0.0, 10.0, 10.0]])
    vels = np.zeros((1, 2))
    prev_obs = np.full((1, 5), -1.0)

    gate = np.array([[False]])
    matched, unmatched_dets, unmatched_trks = associate(
        dets, trks, vels, prev_obs,
        iou_threshold=0.3, inertia_weight=0.2, gate_mask=gate,
    )
    assert matched.shape[0] == 0
    assert list(unmatched_dets) == [0]
    assert list(unmatched_trks) == [0]


def test_associate_gate_mask_allows_match():
    """A permissive gate must not change an obvious match."""
    dets = np.array([[0.0, 0.0, 10.0, 10.0]])
    trks = np.array([[1.0, 1.0, 11.0, 11.0]])
    vels = np.zeros((1, 2))
    prev_obs = np.full((1, 5), -1.0)

    gate = np.array([[True]])
    matched, _, _ = associate(
        dets, trks, vels, prev_obs,
        iou_threshold=0.3, inertia_weight=0.2, gate_mask=gate,
    )
    assert matched.shape[0] == 1
    assert matched[0, 0] == 0 and matched[0, 1] == 0


def test_associate_gate_mask_reroutes_assignment():
    """When the best pair is gated out, the solver must pick the allowed one."""
    # Det 0 overlaps both tracks; track 0 slightly better spatially,
    # but the gate forbids det0-trk0.
    dets = np.array([[0.0, 0.0, 10.0, 10.0]])
    trks = np.array([
        [0.0, 0.0, 10.0, 10.0],   # IoU 1.0 but gated out
        [2.0, 2.0, 12.0, 12.0],   # lower IoU, allowed
    ])
    vels = np.zeros((2, 2))
    prev_obs = np.full((2, 5), -1.0)

    gate = np.array([[False, True]])
    matched, _, unmatched_trks = associate(
        dets, trks, vels, prev_obs,
        iou_threshold=0.3, inertia_weight=0.2, gate_mask=gate,
    )
    assert matched.shape[0] == 1
    assert matched[0, 1] == 1
    assert 0 in unmatched_trks


def test_associate_gate_mask_none_is_noop():
    """gate_mask=None must behave exactly like the ungated call."""
    dets = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [50.0, 50.0, 60.0, 60.0],
    ])
    trks = np.array([
        [1.0, 1.0, 11.0, 11.0],
        [51.0, 51.0, 61.0, 61.0],
    ])
    vels = np.zeros((2, 2))
    prev_obs = np.full((2, 5), -1.0)

    m_none, ud_none, ut_none = associate(
        dets, trks, vels, prev_obs, iou_threshold=0.3, inertia_weight=0.2,
    )
    m_gate, ud_gate, ut_gate = associate(
        dets, trks, vels, prev_obs, iou_threshold=0.3, inertia_weight=0.2,
        gate_mask=np.ones((2, 2), dtype=bool),
    )
    np.testing.assert_array_equal(m_none, m_gate)
    np.testing.assert_array_equal(ud_none, ud_gate)
    np.testing.assert_array_equal(ut_none, ut_gate)


# ---------------------------------------------------------------------------
# emb_rescue_thresh -- appearance rescue for zero-IoU pairs inside the gate
# ---------------------------------------------------------------------------

def _rescue_scene():
    """One detection with zero IoU to the single track prediction."""
    dets = np.array([[100.0, 0.0, 120.0, 40.0]])
    trks = np.array([[0.0, 0.0, 20.0, 40.0]])
    vels = np.zeros((1, 2))
    prev_obs = np.full((1, 5), -1.0)
    return dets, trks, vels, prev_obs


def test_emb_rescue_matches_zero_iou_pair():
    """High appearance similarity inside the gate rescues a zero-IoU pair."""
    dets, trks, vels, prev_obs = _rescue_scene()
    matched, _, _ = associate(
        dets, trks, vels, prev_obs,
        iou_threshold=0.3, inertia_weight=0.2,
        emb_cost=np.array([[0.9]]),
        gate_mask=np.array([[True]]),
        emb_rescue_thresh=0.6,
    )
    assert matched.shape[0] == 1


def test_emb_rescue_rejects_low_similarity():
    """Appearance below the rescue threshold must not rescue."""
    dets, trks, vels, prev_obs = _rescue_scene()
    matched, unmatched_dets, _ = associate(
        dets, trks, vels, prev_obs,
        iou_threshold=0.3, inertia_weight=0.2,
        emb_cost=np.array([[0.4]]),
        gate_mask=np.array([[True]]),
        emb_rescue_thresh=0.6,
    )
    assert matched.shape[0] == 0
    assert list(unmatched_dets) == [0]


def test_emb_rescue_requires_gate_pass():
    """Rescue must never fire outside the motion gate."""
    dets, trks, vels, prev_obs = _rescue_scene()
    matched, _, _ = associate(
        dets, trks, vels, prev_obs,
        iou_threshold=0.3, inertia_weight=0.2,
        emb_cost=np.array([[0.99]]),
        gate_mask=np.array([[False]]),
        emb_rescue_thresh=0.6,
    )
    assert matched.shape[0] == 0


def test_emb_rescue_disabled_without_gate_mask():
    """Without a motion gate, rescue is unbounded and must stay off."""
    dets, trks, vels, prev_obs = _rescue_scene()
    matched, _, _ = associate(
        dets, trks, vels, prev_obs,
        iou_threshold=0.3, inertia_weight=0.2,
        emb_cost=np.array([[0.99]]),
        gate_mask=None,
        emb_rescue_thresh=0.6,
    )
    assert matched.shape[0] == 0


def test_emb_rescue_none_preserves_masking():
    """Default (None) keeps the original behavior: appearance cannot
    rescue non-overlapping pairs."""
    dets, trks, vels, prev_obs = _rescue_scene()
    matched, _, _ = associate(
        dets, trks, vels, prev_obs,
        iou_threshold=0.3, inertia_weight=0.2,
        emb_cost=np.array([[0.99]]),
        gate_mask=np.array([[True]]),
        emb_rescue_thresh=None,
    )
    assert matched.shape[0] == 0


def test_emb_rescue_prefers_overlapping_pair():
    """With two candidates, the solver must still prefer the pair with
    both spatial and appearance support over a rescue-only pair."""
    dets = np.array([[0.0, 0.0, 20.0, 40.0]])
    trks = np.array([
        [2.0, 0.0, 22.0, 40.0],     # overlapping track
        [100.0, 0.0, 120.0, 40.0],  # distant track, same appearance
    ])
    vels = np.zeros((2, 2))
    prev_obs = np.full((2, 5), -1.0)
    matched, _, _ = associate(
        dets, trks, vels, prev_obs,
        iou_threshold=0.3, inertia_weight=0.2,
        emb_cost=np.array([[0.9, 0.9]]),
        gate_mask=np.array([[True, True]]),
        emb_rescue_thresh=0.6,
    )
    assert matched.shape[0] == 1
    assert matched[0, 1] == 0


# ---------------------------------------------------------------------------
# hybrid fallback_cap plumbing through associate()
# ---------------------------------------------------------------------------

def test_associate_hybrid_fallback_cap_blocks_nonoverlap_match():
    """With fallback_cap below the threshold, a zero-IoU nearby pair
    must no longer match in stage 1."""
    dets = np.array([[10.5, 0.0, 20.5, 40.0]])   # nearby, zero IoU
    trks = np.array([[0.0, 0.0, 10.0, 40.0]])
    vels = np.zeros((1, 2))
    prev_obs = np.full((1, 5), -1.0)

    # Default cap (0.5): fallback can clear iou_threshold=0.3 and match
    matched_default, _, _ = associate(
        dets, trks, vels, prev_obs,
        iou_threshold=0.3, inertia_weight=0.2,
        use_hybrid_distance=True, hybrid_config={},
    )
    assert matched_default.shape[0] == 1

    # Capped below the threshold: same pair must be rejected
    matched_capped, _, _ = associate(
        dets, trks, vels, prev_obs,
        iou_threshold=0.3, inertia_weight=0.2,
        use_hybrid_distance=True,
        hybrid_config={"fallback_cap": 0.25},
    )
    assert matched_capped.shape[0] == 0


# ---------------------------------------------------------------------------
# _compute_adaptive_weight vectorization -- equivalence with reference loop
# ---------------------------------------------------------------------------

def _adaptive_weight_reference(emb_cost, w_assoc_emb, bottom=0.5):
    """Original loop implementation, kept as the behavioral oracle."""
    w_emb = np.full_like(emb_cost, w_assoc_emb)
    if bottom >= 1.0:
        return w_emb * emb_cost
    for idx in range(emb_cost.shape[0]):
        inds = np.argsort(-emb_cost[idx])
        if len(inds) < 2:
            continue
        best = emb_cost[idx, inds[0]]
        if best == 0:
            w_emb[idx] *= 0.0
        else:
            second = emb_cost[idx, inds[1]]
            ratio = second / best
            w_emb[idx] *= 1.0 - max((ratio - bottom), 0.0) / (1.0 - bottom)
    for idj in range(emb_cost.shape[1]):
        inds = np.argsort(-emb_cost[:, idj])
        if len(inds) < 2:
            continue
        best = emb_cost[inds[0], idj]
        if best == 0:
            w_emb[:, idj] *= 0.0
        else:
            second = emb_cost[inds[1], idj]
            ratio = second / best
            w_emb[:, idj] *= 1.0 - max((ratio - bottom), 0.0) / (1.0 - bottom)
    return w_emb * emb_cost


def test_adaptive_weight_matches_reference_random():
    from arcsort.association import _compute_adaptive_weight

    rng = np.random.RandomState(42)
    for shape in [(1, 1), (1, 5), (5, 1), (2, 2), (7, 3), (10, 10)]:
        emb = rng.uniform(-1.0, 1.0, size=shape)
        for bottom in [0.0, 0.3, 0.5, 0.9, 1.0]:
            got = _compute_adaptive_weight(emb.copy(), 1.25, bottom=bottom)
            want = _adaptive_weight_reference(emb.copy(), 1.25, bottom=bottom)
            np.testing.assert_allclose(got, want, atol=1e-12,
                                       err_msg=f"shape={shape} bottom={bottom}")


def test_adaptive_weight_matches_reference_with_zeros():
    """Zeroed rows/columns (gated ReID) must reproduce the best==0 branch."""
    from arcsort.association import _compute_adaptive_weight

    emb = np.array([
        [0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.5, 0.5, 0.0],
    ])
    got = _compute_adaptive_weight(emb.copy(), 1.25, bottom=0.5)
    want = _adaptive_weight_reference(emb.copy(), 1.25, bottom=0.5)
    np.testing.assert_allclose(got, want, atol=1e-12)
