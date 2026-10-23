"""Tests for arcsort/matching.py -- IoU computation and linear assignment."""

from __future__ import annotations

import numpy as np
import pytest

from arcsort.matching import hybrid_score_batch, iou_batch, linear_assignment


# ---------------------------------------------------------------------------
# iou_batch
# ---------------------------------------------------------------------------

def test_iou_batch_identical_boxes():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0]])
    iou = iou_batch(a, b)
    assert iou.shape == (1, 1)
    assert iou[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_iou_batch_no_overlap():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[20.0, 20.0, 30.0, 30.0]])
    iou = iou_batch(a, b)
    assert iou[0, 0] == pytest.approx(0.0, abs=1e-5)


def test_iou_batch_partial_overlap():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[5.0, 5.0, 15.0, 15.0]])
    iou = iou_batch(a, b)
    # intersection = 5*5 = 25, union = 100+100-25 = 175
    assert iou[0, 0] == pytest.approx(25.0 / 175.0, abs=1e-5)


def test_iou_batch_contained_box():
    a = np.array([[0.0, 0.0, 20.0, 20.0]])
    b = np.array([[5.0, 5.0, 15.0, 15.0]])
    iou = iou_batch(a, b)
    # intersection = 100, union = 400+100-100 = 400
    assert iou[0, 0] == pytest.approx(100.0 / 400.0, abs=1e-5)


def test_iou_batch_multiple_boxes():
    a = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [50.0, 50.0, 60.0, 60.0],
    ])
    b = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [50.0, 50.0, 60.0, 60.0],
    ])
    iou = iou_batch(a, b)
    assert iou.shape == (2, 2)
    assert iou[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert iou[1, 1] == pytest.approx(1.0, abs=1e-5)
    assert iou[0, 1] == pytest.approx(0.0, abs=1e-5)
    assert iou[1, 0] == pytest.approx(0.0, abs=1e-5)


def test_iou_batch_empty_a():
    a = np.empty((0, 4))
    b = np.array([[0.0, 0.0, 10.0, 10.0]])
    iou = iou_batch(a, b)
    assert iou.shape == (0, 1)


def test_iou_batch_empty_b():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.empty((0, 4))
    iou = iou_batch(a, b)
    assert iou.shape == (1, 0)


def test_iou_batch_both_empty():
    a = np.empty((0, 4))
    b = np.empty((0, 4))
    iou = iou_batch(a, b)
    assert iou.shape == (0, 0)


def test_iou_batch_single_each():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0]])
    iou = iou_batch(a, b)
    assert iou.shape == (1, 1)
    assert iou[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_iou_batch_asymmetric_sizes():
    a = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [20.0, 20.0, 30.0, 30.0],
        [40.0, 40.0, 50.0, 50.0],
    ])
    b = np.array([
        [0.0, 0.0, 10.0, 10.0],
    ])
    iou = iou_batch(a, b)
    assert iou.shape == (3, 1)
    assert iou[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert iou[1, 0] == pytest.approx(0.0, abs=1e-5)
    assert iou[2, 0] == pytest.approx(0.0, abs=1e-5)


def test_iou_batch_touching_edge():
    """Boxes that share an edge have zero area intersection."""
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[10.0, 0.0, 20.0, 10.0]])
    iou = iou_batch(a, b)
    assert iou[0, 0] == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# linear_assignment
# ---------------------------------------------------------------------------

def test_linear_assignment_simple():
    cost = np.array([
        [1.0, 10.0],
        [10.0, 1.0],
    ])
    matches = linear_assignment(cost)
    assert matches.shape[1] == 2
    # Optimal: row 0->col 0, row 1->col 1 (total cost=2)
    match_set = {(r, c) for r, c in matches}
    assert (0, 0) in match_set
    assert (1, 1) in match_set


def test_linear_assignment_empty():
    cost = np.empty((0, 0))
    matches = linear_assignment(cost)
    assert matches.shape == (0, 2)


def test_linear_assignment_single_element():
    cost = np.array([[5.0]])
    matches = linear_assignment(cost)
    assert matches.shape == (1, 2)
    assert matches[0, 0] == 0
    assert matches[0, 1] == 0


def test_linear_assignment_rectangular_more_rows():
    cost = np.array([
        [1.0, 10.0],
        [10.0, 1.0],
        [5.0, 5.0],
    ])
    matches = linear_assignment(cost)
    # At most 2 matches (min of 3 rows, 2 cols)
    assert matches.shape[0] <= 2


def test_linear_assignment_rectangular_more_cols():
    cost = np.array([
        [1.0, 10.0, 5.0],
        [10.0, 1.0, 5.0],
    ])
    matches = linear_assignment(cost)
    assert matches.shape[0] <= 2


def test_linear_assignment_finds_optimal():
    """Verify that the assignment minimizes total cost."""
    cost = np.array([
        [4.0, 1.0, 3.0],
        [2.0, 0.0, 5.0],
        [3.0, 2.0, 2.0],
    ])
    matches = linear_assignment(cost)
    total = sum(cost[r, c] for r, c in matches)
    # Optimal: (0,1)=1 + (1,0)=2 + (2,2)=2 = 5
    assert total == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# hybrid_score_batch
# ---------------------------------------------------------------------------

def test_hybrid_score_overlapping_uses_iou():
    """When IoU > fallback threshold, hybrid should return IoU values."""
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[1.0, 1.0, 11.0, 11.0]])
    iou = iou_batch(a, b)
    hybrid = hybrid_score_batch(a, b, iou_fallback_thresh=0.1)
    np.testing.assert_array_almost_equal(hybrid, iou)


def test_hybrid_score_non_overlapping_uses_fallback():
    """When IoU = 0, hybrid should still produce a positive score if close."""
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[11.0, 0.0, 21.0, 10.0]])  # 1px gap, same size
    iou = iou_batch(a, b)
    assert iou[0, 0] < 0.01  # nearly zero IoU

    hybrid = hybrid_score_batch(a, b, iou_fallback_thresh=0.1)
    # Fallback score is scaled by 0.5, so max ~0.5; should still be positive
    assert hybrid[0, 0] > 0.01, (
        f"Hybrid score {hybrid[0, 0]:.3f} should be >0 for nearby boxes"
    )


def test_hybrid_score_far_away_is_low():
    """Very distant non-overlapping boxes should get near-zero score."""
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[500.0, 500.0, 510.0, 510.0]])
    hybrid = hybrid_score_batch(a, b, iou_fallback_thresh=0.1)
    assert hybrid[0, 0] < 0.01


def test_hybrid_score_identical_boxes():
    """Identical boxes should get score = 1.0 (perfect IoU)."""
    a = np.array([[10.0, 10.0, 50.0, 50.0]])
    b = np.array([[10.0, 10.0, 50.0, 50.0]])
    hybrid = hybrid_score_batch(a, b)
    assert abs(hybrid[0, 0] - 1.0) < 0.01


def test_hybrid_score_shape_penalty():
    """Nearby boxes with very different aspect ratios should be penalized."""
    a = np.array([[0.0, 0.0, 10.0, 100.0]])   # tall
    b = np.array([[12.0, 0.0, 112.0, 10.0]])   # wide, nearby
    hybrid = hybrid_score_batch(a, b, iou_fallback_thresh=0.1)
    # Should be low due to shape mismatch
    assert hybrid[0, 0] < 0.5


def test_hybrid_score_empty_inputs():
    a = np.empty((0, 4))
    b = np.array([[0.0, 0.0, 10.0, 10.0]])
    result = hybrid_score_batch(a, b)
    assert result.shape == (0, 1)


def test_hybrid_score_batch_shape():
    """Output shape should be (N, M)."""
    a = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60]])
    b = np.array([[1, 1, 11, 11], [100, 100, 110, 110]])
    result = hybrid_score_batch(a, b)
    assert result.shape == (3, 2)


# ---------------------------------------------------------------------------
# mahalanobis_distance_batch
# ---------------------------------------------------------------------------

def test_mahalanobis_identity_cov_is_euclidean():
    """With identity covariance, d^2 equals squared Euclidean distance."""
    from arcsort.matching import mahalanobis_distance_batch

    dets = np.array([[3.0, 4.0, 10.0, 10.0]])
    means = np.array([[0.0, 0.0, 10.0, 10.0]])
    covs = np.eye(4)[np.newaxis, :, :]
    d2 = mahalanobis_distance_batch(dets, means, covs)
    assert d2.shape == (1, 1)
    assert d2[0, 0] == pytest.approx(25.0, abs=1e-8)


def test_mahalanobis_scales_with_covariance():
    """Larger covariance shrinks the distance for the same offset."""
    from arcsort.matching import mahalanobis_distance_batch

    dets = np.array([[10.0, 0.0, 10.0, 10.0]])
    means = np.array([[0.0, 0.0, 10.0, 10.0]])
    tight = np.eye(4)[np.newaxis, :, :]
    wide = (np.eye(4) * 100.0)[np.newaxis, :, :]
    d2_tight = mahalanobis_distance_batch(dets, means, tight)
    d2_wide = mahalanobis_distance_batch(dets, means, wide)
    assert d2_wide[0, 0] < d2_tight[0, 0]
    assert d2_wide[0, 0] == pytest.approx(1.0, abs=1e-8)


def test_mahalanobis_singular_cov_fails_open():
    """A singular covariance must yield zero distance (never gated out)."""
    from arcsort.matching import mahalanobis_distance_batch

    dets = np.array([[100.0, 100.0, 10.0, 10.0]])
    means = np.array([[0.0, 0.0, 10.0, 10.0]])
    covs = np.zeros((1, 4, 4))
    d2 = mahalanobis_distance_batch(dets, means, covs)
    assert d2[0, 0] == 0.0


def test_mahalanobis_per_track_covariance():
    """Each column must use its own track's covariance."""
    from arcsort.matching import mahalanobis_distance_batch

    dets = np.array([[10.0, 0.0, 10.0, 10.0]])
    means = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [0.0, 0.0, 10.0, 10.0],
    ])
    covs = np.stack([np.eye(4), np.eye(4) * 4.0])
    d2 = mahalanobis_distance_batch(dets, means, covs)
    assert d2[0, 0] == pytest.approx(100.0, abs=1e-8)
    assert d2[0, 1] == pytest.approx(25.0, abs=1e-8)


def test_mahalanobis_empty_inputs():
    from arcsort.matching import mahalanobis_distance_batch

    d2 = mahalanobis_distance_batch(
        np.empty((0, 4)), np.empty((0, 4)), np.empty((0, 4, 4))
    )
    assert d2.shape == (0, 0)


# ---------------------------------------------------------------------------
# hybrid_score_batch fallback_cap
# ---------------------------------------------------------------------------

def test_hybrid_fallback_cap_default_unchanged():
    """Default cap stays 0.5 so tuned configs reproduce exactly."""
    a = np.array([[0.0, 0.0, 10.0, 20.0]])
    b = np.array([[10.5, 0.0, 20.5, 20.0]])  # nearby, zero IoU
    default = hybrid_score_batch(a, b)
    explicit = hybrid_score_batch(a, b, fallback_cap=0.5)
    assert default[0, 0] == pytest.approx(explicit[0, 0], abs=1e-12)
    assert default[0, 0] > 0.0


def test_hybrid_fallback_cap_bounds_score():
    """Fallback score must never exceed the configured cap."""
    a = np.array([[0.0, 0.0, 10.0, 20.0]])
    b = np.array([[10.5, 0.0, 20.5, 20.0]])
    capped = hybrid_score_batch(a, b, fallback_cap=0.2)
    assert 0.0 < capped[0, 0] <= 0.2


def test_hybrid_fallback_cap_does_not_affect_iou_region():
    """Overlapping pairs use raw IoU regardless of the cap."""
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[1.0, 1.0, 11.0, 11.0]])
    uncapped = hybrid_score_batch(a, b, fallback_cap=0.5)
    capped = hybrid_score_batch(a, b, fallback_cap=0.05)
    assert uncapped[0, 0] == pytest.approx(capped[0, 0], abs=1e-12)


# ---------------------------------------------------------------------------
# eiou_batch -- Expansion IoU with per-track factors
# ---------------------------------------------------------------------------

def test_eiou_zero_expansion_equals_iou():
    from arcsort.matching import eiou_batch

    a = np.array([[0.0, 0.0, 10.0, 20.0], [30.0, 0.0, 45.0, 25.0]])
    b = np.array([[2.0, 1.0, 12.0, 21.0], [100.0, 100.0, 120.0, 130.0]])
    eiou = eiou_batch(a, b, np.zeros(2))
    np.testing.assert_allclose(eiou, iou_batch(a, b), atol=1e-10)


def test_eiou_expansion_bridges_gap():
    """Non-overlapping nearby boxes gain overlap under expansion."""
    from arcsort.matching import eiou_batch

    a = np.array([[12.0, 0.0, 22.0, 20.0]])  # 2px right of b
    b = np.array([[0.0, 0.0, 10.0, 20.0]])
    assert iou_batch(a, b)[0, 0] == 0.0
    assert eiou_batch(a, b, np.array([0.5]))[0, 0] > 0.0


def test_eiou_per_column_expansion():
    """Each track column must use its own expansion factor."""
    from arcsort.matching import eiou_batch

    a = np.array([[12.0, 0.0, 22.0, 20.0]])
    b = np.array([
        [0.0, 0.0, 10.0, 20.0],   # e=0 -> no overlap
        [0.0, 0.0, 10.0, 20.0],   # e=0.5 -> overlap
    ])
    eiou = eiou_batch(a, b, np.array([0.0, 0.5]))
    assert eiou[0, 0] == 0.0
    assert eiou[0, 1] > 0.0


def test_eiou_identical_boxes_stay_one():
    from arcsort.matching import eiou_batch

    a = np.array([[0.0, 0.0, 10.0, 20.0]])
    eiou = eiou_batch(a, a.copy(), np.array([0.7]))
    assert eiou[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_eiou_empty_inputs():
    from arcsort.matching import eiou_batch

    out = eiou_batch(np.empty((0, 4)), np.array([[0.0, 0.0, 10.0, 10.0]]),
                     np.array([0.7]))
    assert out.shape == (0, 1)
