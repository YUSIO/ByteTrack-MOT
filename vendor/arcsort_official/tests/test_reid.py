"""Tests for ReID integration.

Tests embedding similarity, EMA embedding updates, adaptive weighting,
association with ReID fusion, IMM gating logic, and uncertainty weighting.
These tests do NOT require GPU or FastReID weights -- they test the
mathematical/algorithmic components using synthetic embeddings.
"""

from __future__ import annotations

import numpy as np
import pytest

from arcsort.association import _compute_adaptive_weight, associate
from arcsort.matching import embedding_similarity_batch
from arcsort.track import STrack, reset_track_id


@pytest.fixture(autouse=True)
def _reset_ids():
    reset_track_id()
    yield
    reset_track_id()


# ---------------------------------------------------------------------------
# embedding_similarity_batch
# ---------------------------------------------------------------------------

def test_embedding_similarity_identical():
    """Identical L2-normalized embeddings should have similarity 1.0."""
    embs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    sim = embedding_similarity_batch(embs, embs)
    np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-6)


def test_embedding_similarity_orthogonal():
    """Orthogonal embeddings should have similarity 0.0."""
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    sim = embedding_similarity_batch(a, b)
    np.testing.assert_allclose(sim[0, 0], 0.0, atol=1e-6)


def test_embedding_similarity_shape():
    """Output shape should be (N_dets, M_tracks)."""
    dets = np.random.randn(5, 128)
    dets /= np.linalg.norm(dets, axis=1, keepdims=True)
    trks = np.random.randn(3, 128)
    trks /= np.linalg.norm(trks, axis=1, keepdims=True)
    sim = embedding_similarity_batch(dets, trks)
    assert sim.shape == (5, 3)


def test_embedding_similarity_empty_dets():
    """Empty detection embeddings should return empty matrix."""
    dets = np.empty((0, 128))
    trks = np.random.randn(3, 128)
    sim = embedding_similarity_batch(dets, trks)
    assert sim.shape == (0, 3)


def test_embedding_similarity_empty_tracks():
    """Empty track embeddings should return empty matrix."""
    dets = np.random.randn(5, 128)
    trks = np.empty((0, 128))
    sim = embedding_similarity_batch(dets, trks)
    assert sim.shape == (5, 0)


def test_embedding_similarity_range():
    """L2-normalized dot products should be in [-1, 1]."""
    rng = np.random.default_rng(42)
    dets = rng.standard_normal((10, 64))
    dets /= np.linalg.norm(dets, axis=1, keepdims=True)
    trks = rng.standard_normal((8, 64))
    trks /= np.linalg.norm(trks, axis=1, keepdims=True)
    sim = embedding_similarity_batch(dets, trks)
    assert np.all(sim >= -1.0 - 1e-6)
    assert np.all(sim <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# STrack EMA embedding
# ---------------------------------------------------------------------------

def test_strack_emb_initialized_none():
    """New track should have None embedding before update."""
    track = STrack(np.array([0, 0, 10, 10], dtype=float), 0.9)
    assert track.get_emb() is None


def test_strack_update_emb_first():
    """First embedding update should set emb directly (alpha ignored)."""
    track = STrack(np.array([0, 0, 10, 10], dtype=float), 0.9)
    emb = np.array([1.0, 0.0, 0.0])
    track.update_emb(emb, alpha=0.95)
    np.testing.assert_array_equal(track.get_emb(), emb)


def test_strack_update_emb_ema():
    """Subsequent updates should EMA-blend and re-normalize."""
    track = STrack(np.array([0, 0, 10, 10], dtype=float), 0.9)
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([0.0, 1.0, 0.0])
    track.update_emb(emb1, alpha=0.0)  # set to emb1
    track.update_emb(emb2, alpha=0.5)  # blend: 0.5*emb1 + 0.5*emb2
    result = track.get_emb()
    # After normalization, should be roughly [0.707, 0.707, 0]
    expected = np.array([0.5, 0.5, 0.0])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_strack_update_emb_high_alpha_keeps_old():
    """High alpha should heavily weight the old embedding."""
    track = STrack(np.array([0, 0, 10, 10], dtype=float), 0.9)
    emb1 = np.array([1.0, 0.0])
    emb2 = np.array([0.0, 1.0])
    track.update_emb(emb1, alpha=0.0)
    track.update_emb(emb2, alpha=0.99)  # almost all old
    result = track.get_emb()
    # Should be close to emb1 direction
    assert result[0] > result[1]


def test_strack_update_emb_normalized():
    """EMA result should always be L2-normalized."""
    rng = np.random.default_rng(123)
    track = STrack(np.array([0, 0, 10, 10], dtype=float), 0.9)
    for _ in range(10):
        emb = rng.standard_normal(64)
        emb /= np.linalg.norm(emb)
        track.update_emb(emb, alpha=0.9)
    result = track.get_emb()
    np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# _compute_adaptive_weight
# ---------------------------------------------------------------------------

def test_aw_unambiguous_full_weight():
    """When best match is much better than 2nd, weight should be ~full."""
    emb_cost = np.array([[0.9, 0.1]])
    result = _compute_adaptive_weight(emb_cost, w_assoc_emb=1.0, bottom=0.5)
    # 2nd/1st = 0.1/0.9 ~ 0.11, below bottom=0.5, so weight = 1.0
    assert result[0, 0] > 0.8 * emb_cost[0, 0]


def test_aw_ambiguous_reduced_weight():
    """When best and 2nd best are similar, weight should be reduced."""
    emb_cost = np.array([[0.8, 0.79]])
    result = _compute_adaptive_weight(emb_cost, w_assoc_emb=1.0, bottom=0.5)
    # 2nd/1st ≈ 0.99, close to 1.0, weight should be small
    assert result[0, 0] < 0.2 * emb_cost[0, 0]


def test_aw_single_match_keeps_weight():
    """With only one column (one track), weight should stay at base."""
    emb_cost = np.array([[0.8]])
    result = _compute_adaptive_weight(emb_cost, w_assoc_emb=0.75, bottom=0.5)
    # Single element -- no ambiguity possible
    np.testing.assert_allclose(result[0, 0], 0.75 * 0.8, atol=1e-6)


def test_aw_zero_best_gives_zero():
    """If best match is zero, weight should be zero."""
    emb_cost = np.array([[0.0, 0.0]])
    result = _compute_adaptive_weight(emb_cost, w_assoc_emb=1.0, bottom=0.5)
    np.testing.assert_allclose(result, 0.0)


def test_aw_shape_preserved():
    """Output shape should match input."""
    emb_cost = np.random.rand(4, 3)
    result = _compute_adaptive_weight(emb_cost, w_assoc_emb=0.75, bottom=0.5)
    assert result.shape == emb_cost.shape


# ---------------------------------------------------------------------------
# associate with ReID embedding cost
# ---------------------------------------------------------------------------

def _make_velocities(n: int) -> np.ndarray:
    return np.zeros((n, 2))


def _make_prev_obs(bboxes: np.ndarray, conf: float = 0.9) -> np.ndarray:
    n = bboxes.shape[0]
    return np.hstack([bboxes, np.full((n, 1), conf)])


def test_associate_reid_helps_resolve_ambiguity():
    """ReID should break ties when two dets have similar IoU with a track."""
    # Two detections, one track -- both have reasonable IoU
    dets = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [2.0, 2.0, 12.0, 12.0],
    ])
    preds = np.array([[1.0, 1.0, 11.0, 11.0]])
    vels = _make_velocities(1)
    prev = _make_prev_obs(preds)

    # Detection 1 has matching appearance, detection 0 doesn't
    emb_cost = np.array([
        [0.1],   # det 0: low similarity
        [0.9],   # det 1: high similarity
    ])

    matched, unmatched_d, unmatched_t = associate(
        dets, preds, vels, prev,
        iou_threshold=0.1, inertia_weight=0.0,
        emb_cost=emb_cost, w_assoc_emb=1.0, aw_off=True,
    )
    assert matched.shape[0] == 1
    assert matched[0, 0] == 1  # det 1 matched (higher appearance score)
    assert matched[0, 1] == 0  # to track 0


def test_associate_reid_masked_when_no_iou():
    """Embedding cost should be zero when IoU is zero (no spatial overlap)."""
    dets = np.array([[0.0, 0.0, 10.0, 10.0]])
    preds = np.array([[100.0, 100.0, 110.0, 110.0]])  # no overlap
    vels = _make_velocities(1)
    prev = _make_prev_obs(preds)

    # Very high appearance similarity -- but no spatial overlap
    emb_cost = np.array([[0.99]])

    matched, unmatched_d, unmatched_t = associate(
        dets, preds, vels, prev,
        iou_threshold=0.1, inertia_weight=0.0,
        emb_cost=emb_cost, w_assoc_emb=1.0, aw_off=True,
    )
    # Should NOT match: IoU=0 masks the embedding cost, base score < threshold
    assert matched.shape[0] == 0


def test_associate_reid_none_emb_backward_compat():
    """With emb_cost=None, association should work exactly as before."""
    dets = np.array([[0.0, 0.0, 10.0, 10.0]])
    preds = np.array([[1.0, 1.0, 11.0, 11.0]])
    vels = _make_velocities(1)
    prev = _make_prev_obs(preds)

    matched, _, _ = associate(
        dets, preds, vels, prev,
        iou_threshold=0.3, inertia_weight=0.0,
        emb_cost=None,
    )
    assert matched.shape[0] == 1


def test_associate_reid_with_adaptive_weight():
    """Adaptive weighting should reduce weight for ambiguous matches."""
    dets = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 11.0, 11.0],
    ])
    preds = np.array([
        [0.5, 0.5, 10.5, 10.5],
        [50.0, 50.0, 60.0, 60.0],
    ])
    vels = _make_velocities(2)
    prev = _make_prev_obs(preds)

    # Both dets have similar embedding to track 0 (ambiguous)
    emb_cost = np.array([
        [0.8, 0.1],
        [0.79, 0.1],
    ])

    matched, _, _ = associate(
        dets, preds, vels, prev,
        iou_threshold=0.1, inertia_weight=0.0,
        emb_cost=emb_cost, w_assoc_emb=0.75, aw_off=False, aw_param=0.5,
    )
    # Should still produce matches based on IoU
    assert matched.shape[0] >= 1


# ---------------------------------------------------------------------------
# IMM gating logic
# ---------------------------------------------------------------------------

def test_should_activate_reid_young_track():
    """Young tracks should always activate ReID."""
    from arcsort.reid import should_activate_reid

    track = STrack(np.array([0, 0, 10, 10], dtype=float), 0.9)
    # age=1, young_frames=5 -> always active
    assert should_activate_reid(track, cv_thresh=0.7, young_frames=5) is True


def test_should_activate_reid_non_imm():
    """Non-IMM tracks should always activate ReID."""
    from arcsort.reid import should_activate_reid

    track = STrack(np.array([0, 0, 10, 10], dtype=float), 0.9, use_imm=False)
    # Artificially age the track
    track.age = 20
    assert should_activate_reid(track, cv_thresh=0.7, young_frames=5) is True


def test_should_activate_reid_high_cv_prob():
    """IMM track with high CV probability should skip ReID."""
    from arcsort.reid import should_activate_reid

    track = STrack(
        np.array([0, 0, 100, 200], dtype=float), 0.9,
        use_imm=True, imm_config=None,
    )
    track.age = 20  # old enough

    # Set CV model probability high (CV is model 0)
    track.kf.model_probs = np.array([0.9, 0.05, 0.05])

    assert should_activate_reid(track, cv_thresh=0.7, young_frames=5) is False


def test_should_activate_reid_low_cv_prob():
    """IMM track with low CV probability should activate ReID."""
    from arcsort.reid import should_activate_reid

    track = STrack(
        np.array([0, 0, 100, 200], dtype=float), 0.9,
        use_imm=True, imm_config=None,
    )
    track.age = 20

    # Set CV model probability low
    track.kf.model_probs = np.array([0.3, 0.4, 0.3])

    assert should_activate_reid(track, cv_thresh=0.7, young_frames=5) is True


# ---------------------------------------------------------------------------
# IMM get_combined_covariance
# ---------------------------------------------------------------------------

def test_imm_combined_covariance_shape():
    """Combined covariance should be (8, 8) in common subspace."""
    from arcsort.imm_kalman_filter import COMMON_DIM, IMMKalmanFilter

    kf = IMMKalmanFilter()
    kf.initiate(np.array([100.0, 100.0, 50.0, 100.0]))
    cov = kf.get_combined_covariance()
    assert cov.shape == (COMMON_DIM, COMMON_DIM)


def test_imm_combined_covariance_positive_semidefinite():
    """Combined covariance should be positive semi-definite."""
    from arcsort.imm_kalman_filter import IMMKalmanFilter

    kf = IMMKalmanFilter()
    kf.initiate(np.array([100.0, 100.0, 50.0, 100.0]))
    kf.predict()
    kf.update(np.array([101.0, 101.0, 50.0, 100.0]))
    cov = kf.get_combined_covariance()
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues >= -1e-10)


def test_imm_combined_covariance_symmetric():
    """Combined covariance should be symmetric."""
    from arcsort.imm_kalman_filter import IMMKalmanFilter

    kf = IMMKalmanFilter()
    kf.initiate(np.array([100.0, 100.0, 50.0, 100.0]))
    cov = kf.get_combined_covariance()
    np.testing.assert_allclose(cov, cov.T, atol=1e-10)


# ---------------------------------------------------------------------------
# compute_uncertainty_weights
# ---------------------------------------------------------------------------

def test_imm_adaptive_weights_non_imm():
    """Non-IMM tracks should get base weight."""
    from arcsort.reid import compute_imm_adaptive_weights

    tracks = [
        STrack(np.array([0, 0, 10, 10], dtype=float), 0.9, use_imm=False),
    ]
    tracks[0].age = 20
    weights = compute_imm_adaptive_weights(tracks, base_weight=1.25)
    np.testing.assert_allclose(weights[0], 1.25)


def test_imm_adaptive_weights_empty():
    """Empty track list should return empty array."""
    from arcsort.reid import compute_imm_adaptive_weights

    weights = compute_imm_adaptive_weights([], base_weight=1.25)
    assert weights.shape == (0,)


def test_imm_adaptive_weights_high_cv():
    """Track with high CV prob should get reduced weight."""
    from arcsort.reid import compute_imm_adaptive_weights

    track = STrack(
        np.array([0, 0, 100, 200], dtype=float), 0.9,
        use_imm=True, imm_config=None,
    )
    track.age = 20
    track.kf.model_probs = np.array([0.9, 0.05, 0.05])

    weights = compute_imm_adaptive_weights(
        [track], base_weight=1.25, cv_discount=0.5, young_frames=5
    )
    # w = 1.25 * (1 - 0.5 * 0.9) = 1.25 * 0.55 = 0.6875
    np.testing.assert_allclose(weights[0], 0.6875, atol=1e-6)


def test_imm_adaptive_weights_low_cv():
    """Track with low CV prob (non-linear motion) should get full weight."""
    from arcsort.reid import compute_imm_adaptive_weights

    track = STrack(
        np.array([0, 0, 100, 200], dtype=float), 0.9,
        use_imm=True, imm_config=None,
    )
    track.age = 20
    track.kf.model_probs = np.array([0.1, 0.5, 0.4])

    weights = compute_imm_adaptive_weights(
        [track], base_weight=1.25, cv_discount=0.5, young_frames=5
    )
    # w = 1.25 * (1 - 0.5 * 0.1) = 1.25 * 0.95 = 1.1875
    np.testing.assert_allclose(weights[0], 1.1875, atol=1e-6)


def test_imm_adaptive_weights_young_track():
    """Young tracks should always get full base weight."""
    from arcsort.reid import compute_imm_adaptive_weights

    track = STrack(
        np.array([0, 0, 100, 200], dtype=float), 0.9,
        use_imm=True, imm_config=None,
    )
    # age=1, young_frames=5 -> young
    track.kf.model_probs = np.array([0.99, 0.005, 0.005])

    weights = compute_imm_adaptive_weights(
        [track], base_weight=1.25, cv_discount=0.5, young_frames=5
    )
    # Young track: full weight regardless of CV prob
    np.testing.assert_allclose(weights[0], 1.25)


def test_imm_adaptive_weights_zero_discount():
    """cv_discount=0 should give uniform base weight."""
    from arcsort.reid import compute_imm_adaptive_weights

    track = STrack(
        np.array([0, 0, 100, 200], dtype=float), 0.9,
        use_imm=True, imm_config=None,
    )
    track.age = 20
    track.kf.model_probs = np.array([0.9, 0.05, 0.05])

    weights = compute_imm_adaptive_weights(
        [track], base_weight=1.25, cv_discount=0.0
    )
    np.testing.assert_allclose(weights[0], 1.25)


# ---------------------------------------------------------------------------
# Embedding cache (tools/cache_embeddings.py format)
# ---------------------------------------------------------------------------

def _make_cache_npz(tmp_path, frames: dict[int, tuple[np.ndarray, np.ndarray]]):
    """Write a synthetic embedding cache npz. frames: fid -> (boxes, embs)."""
    arrays = {}
    for fid, (boxes, embs) in frames.items():
        arrays[f"boxes_{fid}"] = boxes.astype(np.float32)
        arrays[f"embs_{fid}"] = embs.astype(np.float16)
    path = tmp_path / "seq.npz"
    np.savez_compressed(path, **arrays)
    return path


def _unit(vec):
    v = np.zeros(2048, dtype=np.float32)
    v[: len(vec)] = vec
    return v / np.linalg.norm(v)


def test_embedding_cache_lookup_order_and_normalization(tmp_path):
    from arcsort.reid import EmbeddingComputer

    boxes = np.array([[0.0, 0.0, 10.0, 20.0], [50.0, 0.0, 60.0, 20.0]])
    embs = np.stack([_unit([1.0, 0.0]), _unit([0.0, 1.0])])
    path = _make_cache_npz(tmp_path, {1: (boxes, embs)})

    computer = EmbeddingComputer()
    assert not computer.has_cache()
    computer.load_cache(path)
    assert computer.has_cache()
    computer.set_frame(1)

    # Request in reversed order -- must return matching rows, unit norm
    out = computer.compute_embeddings(None, boxes[::-1].copy())
    np.testing.assert_allclose(out[0], embs[1], atol=1e-3)
    np.testing.assert_allclose(out[1], embs[0], atol=1e-3)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_embedding_cache_miss_returns_zeros_without_frame(tmp_path):
    from arcsort.reid import EmbeddingComputer

    boxes = np.array([[0.0, 0.0, 10.0, 20.0]])
    embs = np.stack([_unit([1.0, 0.0])])
    path = _make_cache_npz(tmp_path, {1: (boxes, embs)})

    computer = EmbeddingComputer()
    computer.load_cache(path)
    computer.set_frame(1)

    unknown = np.array([[999.0, 999.0, 1010.0, 1020.0]])
    out = computer.compute_embeddings(None, unknown)
    np.testing.assert_allclose(out, 0.0)


def test_embedding_cache_float_noise_tolerant(tmp_path):
    """Boxes matching within 0.05px (rounding to 1 decimal) must hit."""
    from arcsort.reid import EmbeddingComputer

    boxes = np.array([[0.0, 0.0, 10.0, 20.0]])
    embs = np.stack([_unit([1.0, 0.0])])
    path = _make_cache_npz(tmp_path, {3: (boxes, embs)})

    computer = EmbeddingComputer()
    computer.load_cache(path)
    computer.set_frame(3)

    nudged = boxes + 0.04
    out = computer.compute_embeddings(None, nudged)
    np.testing.assert_allclose(out[0], embs[0], atol=1e-3)


def test_tracker_reid_from_cache_without_frames(tmp_path):
    """ARCTracker with use_reid and a loaded cache must run ReID with
    frame=None (no GPU, no images) and keep IDs on two objects."""
    from arcsort.reid import EmbeddingComputer
    from arcsort.tracker import ARCTracker

    box_a = [10.0, 10.0, 30.0, 50.0]
    box_b = [200.0, 10.0, 220.0, 50.0]
    emb_a, emb_b = _unit([1.0, 0.0]), _unit([0.0, 1.0])

    frames = {}
    for fid in range(1, 8):
        shift = 3.0 * (fid - 1)
        boxes = np.array([
            [box_a[0] + shift, box_a[1], box_a[2] + shift, box_a[3]],
            [box_b[0] - shift, box_b[1], box_b[2] - shift, box_b[3]],
        ])
        frames[fid] = (boxes, np.stack([emb_a, emb_b]))
    path = _make_cache_npz(tmp_path, frames)

    tracker = ARCTracker({"use_reid": True, "min_hits": 1})
    tracker.embedding_computer.load_cache(path)

    ids_per_frame = []
    for fid in range(1, 8):
        boxes, _ = frames[fid]
        dets = np.hstack([boxes, np.full((2, 1), 0.9), np.zeros((2, 1))])
        out = tracker.update(dets, frame=None)
        ids_per_frame.append(sorted(int(r[4]) for r in out))

    assert ids_per_frame[-1] == ids_per_frame[0]
    assert len(set(ids_per_frame[-1])) == 2
    # Track embeddings must have been seeded from the cache
    assert all(t.get_emb() is not None for t in tracker.tracks)
