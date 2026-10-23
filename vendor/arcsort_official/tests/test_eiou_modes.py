"""Tests for eiou.mode -- fixed/adaptive/velocity_switch/aniso/sigma."""

from __future__ import annotations

import numpy as np
import pytest

from arcsort.imm_kalman_filter import IMMKalmanFilter
from arcsort.kalman_filter import KalmanFilter
from arcsort.matching import (
    SIGMA_GAMMA,
    SIGMA_W0,
    SIGMA_WI,
    diou_batch,
    eiou_batch,
    iou_batch,
    sigma_diou_batch,
)
from arcsort.track import reset_track_id
from arcsort.tracker import ARCTracker


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
# eiou_batch -- anisotropic (M, 2) expansion factors
# ---------------------------------------------------------------------------

def test_eiou_aniso_equal_axes_matches_scalar_exactly():
    """(M, 2) factors with ex == ey must reproduce the scalar path bitwise."""
    rng = np.random.default_rng(42)
    a = rng.uniform(0.0, 100.0, size=(5, 2))
    a = np.hstack([a, a + rng.uniform(5.0, 30.0, size=(5, 2))])
    b = rng.uniform(0.0, 100.0, size=(4, 2))
    b = np.hstack([b, b + rng.uniform(5.0, 30.0, size=(4, 2))])
    e = np.array([0.0, 0.3, 0.7, 1.2])

    scalar = eiou_batch(a, b, e)
    aniso = eiou_batch(a, b, np.stack([e, e], axis=1))
    np.testing.assert_array_equal(aniso, scalar)


def test_eiou_aniso_width_only_bridges_horizontal_gap():
    """Width expansion bridges a horizontal gap that height expansion
    cannot."""
    a = np.array([[12.0, 0.0, 22.0, 20.0]])  # 2px right of b
    b = np.array([[0.0, 0.0, 10.0, 20.0]])
    assert iou_batch(a, b)[0, 0] == 0.0
    assert eiou_batch(a, b, np.array([[0.5, 0.0]]))[0, 0] > 0.0
    assert eiou_batch(a, b, np.array([[0.0, 0.5]]))[0, 0] == 0.0


def test_eiou_aniso_height_only_bridges_vertical_gap():
    a = np.array([[0.0, 22.0, 10.0, 42.0]])  # 2px below b
    b = np.array([[0.0, 0.0, 10.0, 20.0]])
    assert eiou_batch(a, b, np.array([[0.0, 0.5]]))[0, 0] > 0.0
    assert eiou_batch(a, b, np.array([[0.5, 0.0]]))[0, 0] == 0.0


def test_eiou_aniso_per_column_factors():
    """Each track column must use its own (ex, ey) pair."""
    a = np.array([[12.0, 0.0, 22.0, 20.0]])
    b = np.array([
        [0.0, 0.0, 10.0, 20.0],   # ey only -> no overlap
        [0.0, 0.0, 10.0, 20.0],   # ex only -> overlap
    ])
    eiou = eiou_batch(a, b, np.array([[0.0, 0.5], [0.5, 0.0]]))
    assert eiou[0, 0] == 0.0
    assert eiou[0, 1] > 0.0


# ---------------------------------------------------------------------------
# get_velocity accessors
# ---------------------------------------------------------------------------

def test_kalman_filter_get_velocity():
    kf = KalmanFilter()
    kf.initiate(np.array([50.0, 60.0, 800.0, 0.5]))
    np.testing.assert_allclose(kf.get_velocity(), [0.0, 0.0])

    kf.x[4, 0] = 3.0
    kf.x[5, 0] = -2.0
    vel = kf.get_velocity()
    np.testing.assert_allclose(vel, [3.0, -2.0])
    # Returned array must be a copy, not a view of filter state
    vel[0] = 999.0
    assert kf.x[4, 0] == 3.0


def test_imm_filter_get_velocity_tracks_motion():
    """After observing steady rightward motion the combined-state
    velocity must point right with near-zero vertical component."""
    kf = IMMKalmanFilter()
    kf.initiate(np.array([100.0, 50.0, 20.0, 40.0]))
    for i in range(1, 9):
        kf.predict()
        kf.update(np.array([100.0 + 10.0 * i, 50.0, 20.0, 40.0]))
    v_cx, v_cy = kf.get_velocity()
    assert v_cx > 5.0
    assert abs(v_cy) < 1.0


# ---------------------------------------------------------------------------
# velocity_switch mode
# ---------------------------------------------------------------------------

def test_velocity_switch_slow_and_fast_tracks():
    """A stationary track gets p_slow, a fast-moving one p_fast."""
    tracker = ARCTracker({
        "use_eiou": True,
        "eiou": {"mode": "velocity_switch", "velocity_thresh": 0.1172,
                 "p_slow": 0.5, "p_fast": 0.6},
        "min_hits": 1,
        "match_thresh": 0.1,
    })
    ids = set()
    for i in range(8):
        x = 300.0 + 30.0 * i   # normalized speed 30/20 = 1.5 >> thresh
        out = tracker.update(_dets(
            (10.0, 10.0, 30.0, 50.0),
            (x, 10.0, x + 20.0, 50.0),
        ))
        for r in out:
            ids.add(int(r[4]))
    assert len(ids) == 2  # association held for both objects

    expansions = tracker._compute_track_expansions()
    assert expansions.shape == (2,)
    for idx, track in enumerate(tracker.tracks):
        x1, y1, x2, y2 = track.get_state()
        w = max(float(x2 - x1), 1.0)
        h = max(float(y2 - y1), 1.0)
        v_cx, v_cy = track.kf.get_velocity()
        speed = float(np.hypot(v_cx / w, v_cy / h))
        expected = 0.5 if speed <= 0.1172 else 0.6
        assert expansions[idx] == expected
    # Both branches must actually be exercised
    assert set(expansions.tolist()) == {0.5, 0.6}


def test_velocity_switch_works_with_imm():
    tracker = ARCTracker({
        "use_imm": True,
        "use_eiou": True,
        "eiou": {"mode": "velocity_switch"},
        "min_hits": 1,
        "match_thresh": 0.1,
    })
    for i in range(8):
        x = 100.0 + 30.0 * i
        tracker.update(_dets((x, 10.0, x + 20.0, 50.0)))
    expansions = tracker._compute_track_expansions()
    assert expansions[-1] == tracker.eiou_config["p_fast"]


def test_velocity_switch_config_defaults():
    tracker = ARCTracker({"use_eiou": True, "eiou": {"mode": "velocity_switch"}})
    assert tracker.eiou_config["velocity_thresh"] == 0.1172
    assert tracker.eiou_config["p_slow"] == 0.5
    assert tracker.eiou_config["p_fast"] == 0.6


# ---------------------------------------------------------------------------
# aniso mode
# ---------------------------------------------------------------------------

def _run_imm_scenario(tracker: ARCTracker) -> None:
    """Feed a track and let it coast so the covariance grows."""
    for i in range(6):
        x = 10.0 + 5 * i
        tracker.update(_dets((x, 10.0, x + 20.0, 50.0)))
    for _ in range(4):
        tracker.update(np.empty((0, 6)))


def test_aniso_expansions_shape_and_max_equals_adaptive():
    """Aniso factors are (num, 2); their per-track max must equal the
    scalar adaptive factor exactly (same sigmas, same clipping)."""
    tracker = ARCTracker({
        "use_imm": True,
        "use_eiou": True,
        "eiou": {"mode": "aniso", "adaptive_scale": 2.0,
                 "min_expansion": 0.0, "max_expansion": 5.0},
        "min_hits": 1,
        "track_buffer": 60,
    })
    _run_imm_scenario(tracker)

    exp_aniso = tracker._compute_track_expansions()
    assert exp_aniso.shape == (len(tracker.tracks), 2)

    tracker.eiou_config = {**tracker.eiou_config, "mode": "adaptive"}
    exp_adaptive = tracker._compute_track_expansions()
    np.testing.assert_array_equal(
        np.maximum(exp_aniso[:, 0], exp_aniso[:, 1]), exp_adaptive
    )


def test_aniso_tracks_maintain_ids_end_to_end():
    """Mode aniso must run through associate()/eiou_batch and keep a
    fast-moving track's ID."""
    tracker = ARCTracker({
        "use_imm": True,
        "use_eiou": True,
        "eiou": {"mode": "aniso", "adaptive_scale": 2.0},
        "min_hits": 1,
        "match_thresh": 0.1,
    })
    ids = set()
    for i in range(8):
        x = 10.0 + 25.0 * i
        out = tracker.update(_dets((x, 10.0, x + 20.0, 50.0)))
        for r in out:
            ids.add(int(r[4]))
    assert len(ids) == 1


# ---------------------------------------------------------------------------
# Mode resolution and guards
# ---------------------------------------------------------------------------

def test_mode_absent_with_adaptive_flag_matches_explicit_adaptive():
    """A config without `mode` and adaptive: true must behave exactly
    like an explicit mode: adaptive (legacy backward compatibility)."""
    eiou_common = {"adaptive_scale": 2.0, "min_expansion": 0.2,
                   "max_expansion": 1.2}

    def run(eiou_cfg: dict) -> tuple[str, np.ndarray, list[np.ndarray]]:
        reset_track_id()
        tracker = ARCTracker({
            "use_imm": True,
            "use_eiou": True,
            "eiou": eiou_cfg,
            "min_hits": 1,
            "track_buffer": 60,
        })
        outs = []
        for i in range(6):
            x = 10.0 + 5 * i
            outs.append(tracker.update(_dets((x, 10.0, x + 20.0, 50.0))))
        for _ in range(4):
            outs.append(tracker.update(np.empty((0, 6))))
        return (
            tracker.eiou_config["mode"],
            tracker._compute_track_expansions(),
            outs,
        )

    mode_legacy, exp_legacy, outs_legacy = run(
        {**eiou_common, "adaptive": True}
    )
    mode_explicit, exp_explicit, outs_explicit = run(
        {**eiou_common, "mode": "adaptive"}
    )

    assert mode_legacy == "adaptive"
    assert mode_explicit == "adaptive"
    np.testing.assert_array_equal(exp_legacy, exp_explicit)
    for a, b in zip(outs_legacy, outs_explicit):
        np.testing.assert_array_equal(a, b)


def test_mode_absent_without_adaptive_flag_resolves_fixed():
    tracker = ARCTracker({"use_eiou": True, "min_hits": 1})
    assert tracker.eiou_config["mode"] == "fixed"
    tracker.update(_dets((10.0, 10.0, 30.0, 50.0)))
    np.testing.assert_allclose(tracker._compute_track_expansions(), [0.7])


def test_adaptive_mode_requires_imm():
    tracker = ARCTracker({
        "use_eiou": True, "use_imm": False,
        "eiou": {"mode": "adaptive"},
    })
    assert tracker.eiou_config["mode"] == "fixed"
    assert tracker.eiou_config["adaptive"] is False


def test_aniso_mode_requires_imm():
    tracker = ARCTracker({
        "use_eiou": True, "use_imm": False,
        "eiou": {"mode": "aniso"},
    })
    assert tracker.eiou_config["mode"] == "fixed"


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="eiou.mode"):
        ARCTracker({"use_eiou": True, "eiou": {"mode": "sigma_typo"}})


# ---------------------------------------------------------------------------
# diou_batch -- Distance-IoU kernel
# ---------------------------------------------------------------------------

def test_diou_identical_boxes_is_one():
    a = np.array([[0.0, 0.0, 10.0, 20.0]])
    assert diou_batch(a, a.copy())[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_diou_separated_boxes_negative_and_monotone():
    """Separated boxes score negative, more negative the further apart,
    and stay above -1 (the DIoU range is (-1, 1])."""
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    near = np.array([[20.0, 0.0, 30.0, 10.0]])
    far = np.array([[60.0, 0.0, 70.0, 10.0]])
    d_near = diou_batch(a, near)[0, 0]
    d_far = diou_batch(a, far)[0, 0]
    assert d_near < 0.0
    assert d_far < d_near
    assert d_far > -1.0


def test_diou_matches_manual_formula():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[20.0, 0.0, 30.0, 10.0]])
    # IoU 0; centers (5,5) vs (25,5): d2 = 400
    # enclosing box (0,0,30,10): c2 = 900 + 100
    expected = 0.0 - 400.0 / (1000.0 + 1e-6)
    assert diou_batch(a, b)[0, 0] == pytest.approx(expected, abs=1e-12)


def test_diou_empty_inputs():
    out = diou_batch(np.empty((0, 4)), np.array([[0.0, 0.0, 10.0, 10.0]]))
    assert out.shape == (0, 1)


# ---------------------------------------------------------------------------
# sigma_diou_batch -- expected DIoU at sigma points
# ---------------------------------------------------------------------------

def test_sigma_weights_are_canonical_ut():
    """Canonical UT for n=2, kappa=1: weights sum exactly to 1."""
    assert SIGMA_W0 + 4 * SIGMA_WI == 1.0
    assert SIGMA_GAMMA == pytest.approx(np.sqrt(3.0), abs=0.0)


def test_sigma_zero_cov_reduces_to_plain_diou():
    """With zero covariance every sigma point collapses onto chi_0, so
    the sigma sum must equal diou_batch bit-for-bit (delta form)."""
    rng = np.random.default_rng(11)
    a = rng.uniform(0.0, 100.0, size=(6, 2))
    a = np.hstack([a, a + rng.uniform(5.0, 30.0, size=(6, 2))])
    b = rng.uniform(0.0, 100.0, size=(5, 2))
    b = np.hstack([b, b + rng.uniform(5.0, 30.0, size=(5, 2))])
    covs = np.zeros((5, 2, 2))

    expected = diou_batch(a, b)
    np.testing.assert_array_equal(sigma_diou_batch(a, b, covs), expected)
    np.testing.assert_array_equal(
        sigma_diou_batch(a, b, covs, pregate=False), expected
    )


def test_sigma_manual_weighted_sum():
    """x-only variance: chi_{1,2} at +-gamma*sigma_x, chi_{3,4} collapse
    onto chi_0. Pins sigma-point construction and the 1/6 weights."""
    a = np.array([[12.0, 0.0, 22.0, 20.0]])
    b = np.array([[0.0, 0.0, 10.0, 20.0]])
    covs = np.array([[[4.0, 0.0], [0.0, 0.0]]])  # sigma_x = 2

    off = np.sqrt(3.0) * 2.0  # gamma * sigma_x, below the offset cap
    shift = np.array([off, 0.0, off, 0.0])
    d0 = diou_batch(a, b)[0, 0]
    dp = diou_batch(a, b + shift)[0, 0]
    dm = diou_batch(a, b - shift)[0, 0]
    expected = d0 + (1.0 / 6.0) * ((dp - d0) + (dm - d0))

    assert sigma_diou_batch(a, b, covs)[0, 0] == pytest.approx(
        expected, rel=1e-12
    )


def test_sigma_symmetry_mirrored_detections():
    """Mirrored detections about the track center must score equally
    (sigma points are +- symmetric)."""
    b = np.array([[0.0, 0.0, 10.0, 20.0]])
    covs = np.array([[[9.0, 0.0], [0.0, 4.0]]])
    right = np.array([[14.0, 0.0, 24.0, 20.0]])
    left = np.array([[-14.0, 0.0, -4.0, 20.0]])

    s_right = sigma_diou_batch(right, b, covs)[0, 0]
    s_left = sigma_diou_batch(left, b, covs)[0, 0]
    assert s_right == pytest.approx(s_left, rel=1e-12)


def test_sigma_pregate_equivalence():
    """Surviving pairs must score identically with and without the
    envelope pre-gate; pre-gated pairs keep the kernel's separated
    value (DIoU at the prediction)."""
    a = np.array([
        [0.0, 0.0, 10.0, 20.0],
        [100.0, 0.0, 110.0, 20.0],
    ])
    b = np.array([
        [2.0, 1.0, 12.0, 21.0],
        [104.0, 2.0, 114.0, 22.0],
        [500.0, 500.0, 520.0, 540.0],
    ])
    covs = np.array([
        [[25.0, 10.0], [10.0, 16.0]],   # correlated
        [[4.0, 0.0], [0.0, 9.0]],
        [[400.0, 0.0], [0.0, 400.0]],
    ])
    max_e = 1.2

    env_mask = eiou_batch(a, b, np.full(3, max_e)) > 0.0
    assert env_mask.any() and (~env_mask).any()

    s_gated = sigma_diou_batch(a, b, covs, max_expansion=max_e)
    s_ungated = sigma_diou_batch(a, b, covs, max_expansion=max_e,
                                 pregate=False)
    np.testing.assert_array_equal(s_gated[env_mask], s_ungated[env_mask])
    np.testing.assert_array_equal(
        s_gated[~env_mask], diou_batch(a, b)[~env_mask]
    )


def test_sigma_degenerate_covariances_are_safe():
    """Zero, huge, and non-PSD covariances (long-LOST tracks) must give
    finite scores in (-1, 1] without raising."""
    a = np.array([[0.0, 0.0, 10.0, 20.0], [30.0, 0.0, 45.0, 25.0]])
    b = np.tile(np.array([[2.0, 1.0, 12.0, 21.0]]), (4, 1))
    covs = np.array([
        [[0.0, 0.0], [0.0, 0.0]],           # zero
        [[1e12, 5e11], [5e11, 1e12]],       # huge (offsets capped)
        [[1.0, 2.0], [2.0, 1.0]],           # non-PSD (negative pivot)
        [[-5.0, 0.0], [0.0, -5.0]],         # negative diagonal
    ])
    s = sigma_diou_batch(a, b, covs)
    assert np.all(np.isfinite(s))
    assert np.all(s <= 1.0)
    assert np.all(s > -1.0)


def test_sigma_saturation_mask():
    """A track beyond the max-expansion envelope of every detection is
    saturated (all plain-IoU sigma evals zero); an overlapping track is
    not. The DIoU kernel still ranks the saturated track."""
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([
        [2.0, 2.0, 12.0, 12.0],
        [500.0, 500.0, 510.0, 510.0],
    ])
    covs = np.array([
        [[4.0, 0.0], [0.0, 4.0]],
        [[1e6, 0.0], [0.0, 1e6]],
    ])
    scores, saturated = sigma_diou_batch(a, b, covs, return_saturation=True)
    np.testing.assert_array_equal(saturated, [False, True])
    assert scores[0, 1] < 0.0


def test_sigma_saturation_with_envelope_overlap():
    """Saturation also fires for a surviving pair whose sigma points all
    miss: envelope overlap does not imply sigma-point overlap."""
    a = np.array([[13.0, 0.0, 23.0, 10.0]])   # 3px gap, inside envelope
    b = np.array([[0.0, 0.0, 10.0, 10.0]])
    covs = np.zeros((1, 2, 2))                # all points at chi_0
    scores, saturated = sigma_diou_batch(a, b, covs, return_saturation=True)
    np.testing.assert_array_equal(saturated, [True])
    np.testing.assert_array_equal(scores, diou_batch(a, b))


# ---------------------------------------------------------------------------
# eiou.mode: sigma -- tracker integration
# ---------------------------------------------------------------------------

def _run_sigma_scenario(tracker: ARCTracker) -> list[np.ndarray]:
    """Two objects, one coasting gap, one re-acquisition."""
    outs = []
    for i in range(6):
        x = 10.0 + 5 * i
        outs.append(tracker.update(_dets(
            (x, 10.0, x + 20.0, 50.0),
            (200.0, 10.0, 220.0, 50.0),
        )))
    for _ in range(3):
        outs.append(tracker.update(np.empty((0, 6))))
    outs.append(tracker.update(_dets(
        (50.0, 10.0, 70.0, 50.0),
        (200.0, 10.0, 220.0, 50.0),
    )))
    return outs


def test_sigma_beta1_reproduces_adaptive_exactly():
    """sigma_blend_beta = 1.0 must reproduce mode "adaptive" bit-for-bit:
    identical per-frame outputs AND identical spatial score matrices."""
    def build(eiou_cfg: dict):
        reset_track_id()
        tracker = ARCTracker({
            "use_imm": True,
            "use_eiou": True,
            "eiou": eiou_cfg,
            "min_hits": 1,
            "track_buffer": 60,
        })
        return tracker, _run_sigma_scenario(tracker)

    t_sigma, outs_sigma = build(
        {"mode": "sigma", "sigma_blend_beta": 1.0, "adaptive_scale": 2.0}
    )
    t_adapt, outs_adapt = build(
        {"mode": "adaptive", "adaptive_scale": 2.0}
    )
    for a, b in zip(outs_sigma, outs_adapt):
        np.testing.assert_array_equal(a, b)

    # Score-matrix exactness on the final tracker state
    dets = _dets((50.0, 10.0, 70.0, 50.0))[:, :4]
    preds = np.array([t.get_state() for t in t_sigma.tracks])
    num = len(t_sigma.tracks)
    means = np.empty((num, 4))
    covs = np.empty((num, 4, 4))
    for idx, track in enumerate(t_sigma.tracks):
        mean, S = track.kf.get_gating_distribution()
        means[idx] = mean
        covs[idx] = S

    score_sigma = t_sigma._compute_sigma_scores(dets, preds, means, covs)
    expansions = t_sigma._compute_track_expansions(
        mode="adaptive", gating_means=means, gating_covs=covs
    )
    score_adaptive = eiou_batch(dets, preds, expansions)
    np.testing.assert_array_equal(score_sigma, score_adaptive)


def test_sigma_mode_requires_imm():
    tracker = ARCTracker({
        "use_eiou": True, "use_imm": False,
        "eiou": {"mode": "sigma"},
    })
    assert tracker.eiou_config["mode"] == "fixed"


def test_sigma_config_defaults_and_stats_init():
    tracker = ARCTracker({
        "use_imm": True, "use_eiou": True, "eiou": {"mode": "sigma"},
    })
    assert tracker.eiou_config["sigma_blend_beta"] == 0.3
    assert tracker.eiou_config["sigma_debug"] is False
    assert tracker.sigma_saturation_stats == {
        "frames": 0, "total_tracks": 0,
        "saturated_tracks": 0, "per_frame": [],
    }


def test_sigma_expansions_helper_rejects_sigma_mode():
    tracker = ARCTracker({
        "use_imm": True, "use_eiou": True, "eiou": {"mode": "sigma"},
    })
    with pytest.raises(ValueError, match="sigma"):
        tracker._compute_track_expansions()


def test_sigma_negative_score_never_matches():
    """A far detection scores negative (DIoU) and must start a new
    track instead of matching -- stage-1 acceptance requires
    score >= match_thresh > 0 (documented no-clamp decision)."""
    tracker = ARCTracker({
        "use_imm": True,
        "use_eiou": True,
        "eiou": {"mode": "sigma", "sigma_blend_beta": 0.0},
        "min_hits": 1,
    })
    tracker.update(_dets((10.0, 10.0, 30.0, 50.0)))
    tracker.update(_dets((500.0, 10.0, 520.0, 50.0)))
    assert len(tracker.tracks) == 2
    assert tracker.tracks[0].time_since_update > 0
    assert tracker.tracks[1].time_since_update == 0


def test_sigma_saturation_stats_accumulate():
    """sigma_debug accumulates per-frame saturation on the tracker."""
    tracker = ARCTracker({
        "use_imm": True,
        "use_eiou": True,
        "eiou": {"mode": "sigma", "sigma_debug": True},
        "min_hits": 1,
    })
    # Frame 1: no tracks yet -> no stats row
    tracker.update(_dets((0.0, 0.0, 10.0, 10.0)))
    assert tracker.sigma_saturation_stats["frames"] == 0

    # Frame 2: one track, far detection -> fully saturated
    tracker.update(_dets((500.0, 500.0, 510.0, 510.0)))
    stats = tracker.sigma_saturation_stats
    assert stats["frames"] == 1
    assert stats["total_tracks"] == 1
    assert stats["saturated_tracks"] == 1
    assert stats["per_frame"] == [1.0]

    # Frame 3: two tracks, det overlaps the new one -> half saturated
    tracker.update(_dets((500.0, 500.0, 510.0, 510.0)))
    assert stats["frames"] == 2
    assert stats["total_tracks"] == 3
    assert stats["saturated_tracks"] == 2
    assert stats["per_frame"][-1] == 0.5

    # reset() clears the accumulator
    tracker.reset()
    assert tracker.sigma_saturation_stats["frames"] == 0
    assert tracker.sigma_saturation_stats["per_frame"] == []


def test_sigma_debug_off_accumulates_nothing():
    tracker = ARCTracker({
        "use_imm": True, "use_eiou": True,
        "eiou": {"mode": "sigma"},
        "min_hits": 1,
    })
    tracker.update(_dets((0.0, 0.0, 10.0, 10.0)))
    tracker.update(_dets((2.0, 0.0, 12.0, 10.0)))
    assert tracker.sigma_saturation_stats["frames"] == 0


def test_sigma_lost_track_end_to_end_robustness():
    """A long-LOST track (large, possibly ill-conditioned S) must not
    break the sigma score path when detections reappear."""
    tracker = ARCTracker({
        "use_imm": True,
        "use_eiou": True,
        "eiou": {"mode": "sigma"},
        "min_hits": 1,
        "track_buffer": 60,
    })
    for i in range(5):
        x = 10.0 + 8 * i
        tracker.update(_dets((x, 10.0, x + 20.0, 50.0)))
    for _ in range(12):
        tracker.update(np.empty((0, 6)))
    out = tracker.update(_dets((150.0, 10.0, 170.0, 50.0)))
    assert np.all(np.isfinite(out))
    assert len(tracker.tracks) >= 1
