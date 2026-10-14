import numpy as np

from fctrack.formation import OutlierFilter, delaunay_adjacency, estimate, fit_similarity, mst_adjacency, select_neighbours


def test_mst_is_a_tree_with_the_shortest_edges():
    P = np.array([[0, 0], [1, 0], [2, 0], [2, 5], [10, 0]], float)
    adj = mst_adjacency(P)
    edges = {tuple(sorted((i, j))) for i in range(len(P)) for j in adj[i]}
    assert edges == {(0, 1), (1, 2), (2, 3), (2, 4)}


def test_neighbours_walk_past_unavailable_members():
    P = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [0, 4]], float)
    avail = np.array([False, False, True, True, True])
    assert select_neighbours(P, 0, avail, "mst", 2) == [4, 2]  # tree neighbours of 0 are 1 (missing itself) and 4; 2 lies behind 1
    assert select_neighbours(P, 0, avail, "knn", 2) == [2, 3]


def test_offset_model_keeps_relative_position_under_translation():
    N0 = np.array([[0, 0], [10, 0], [0, 10]], float)
    c, spread, used = estimate(np.array([4.0, 4.0]), N0, N0 + [3, -2], "offset", eps=1.0)
    assert np.allclose(c, [7, 2]) and spread < 1e-9 and used == "offset"


def test_affine_model_is_exact_under_rotation_scale_and_shear():
    rng = np.random.default_rng(0)
    N0 = rng.uniform(0, 100, (5, 2))
    A, t = np.array([[1.2, 0.3], [-0.2, 0.9]]), np.array([5.0, -7.0])
    p0 = N0.mean(0) + [3, 1]
    c, _, used = estimate(p0, N0, N0 @ A.T + t, "affine", eps=1.0)
    assert used == "affine" and np.allclose(c, A @ p0 + t)
    c, _, _ = estimate(p0, N0, N0 @ A.T + t, "offset", eps=1.0)
    assert not np.allclose(c, A @ p0 + t, atol=0.5)  # the translation model cannot follow the deformation


def test_affine_falls_back_when_neighbours_are_collinear_or_far():
    N0 = np.array([[0, 0], [10, 0], [20, 0]], float)
    assert estimate(np.array([5.0, 3.0]), N0, N0 + 1, "affine", eps=1.0)[2] == "similarity"
    N0 = np.array([[0, 0], [1, 0], [0, 1]], float)
    assert estimate(np.array([50.0, 50.0]), N0, N0 + 1, "affine", eps=1.0)[2] == "similarity"


def test_similarity_fit_recovers_rotation_and_scale():
    A = np.array([[0, 0], [1, 0], [0, 2]], float)
    th = 0.4
    R = 1.5 * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    M, t = fit_similarity(A, A @ R.T + [2, 3])
    assert np.allclose(M, R) and np.allclose(t, [2, 3])


def test_delaunay_adjacency_and_fallback():
    assert delaunay_adjacency(np.array([[0, 0], [1, 0], [2, 0]], float)) is None
    adj = delaunay_adjacency(np.array([[0, 0], [4, 0], [0, 4], [4, 4], [2, 2]], float))
    assert sorted(adj[4]) == [0, 1, 2, 3]


def test_outlier_filter_flags_a_member_that_leaves_the_formation():
    f = OutlierFilter(history=30, percentile=95.0, regulariser=1e-2, min_members=4, min_samples=30)
    base = np.array([[0, 0], [10, 0], [0, 10], [10, 10], [5, 5], [5, 0], [0, 5], [10, 5]], float)
    for _ in range(10):
        assert not f.update(base + np.random.default_rng(1).normal(0, 0.1, base.shape)).any()
    moved = base.copy()
    moved[4] = [60, 60]
    assert f.update(moved)[4]


def test_paper_consistency_uses_the_agreeing_pair():
    # the paper's example: a 40 x 30 box (threshold 120 px^2), estimates (50, 50), (55, 52), (80, 90)
    p0 = np.array([0.0, 0.0])
    N0 = np.array([[-10.0, 0.0], [0.0, -20.0], [30.0, 0.0]])
    each = np.array([[50.0, 50.0], [55.0, 52.0], [80.0, 90.0]])
    c, _, used = estimate(p0, N0, each - (p0 - N0), "offset_paper", eps=0.1 * 40 * 30)
    assert used == "offset_agree" and np.allclose(c, [52.5, 51.0])
    each = np.array([[50.0, 50.0], [70.0, 52.0], [80.0, 90.0]])  # nobody agrees: weighted by 1 / reference distance
    c, _, used = estimate(p0, N0, each - (p0 - N0), "offset_paper", eps=0.1 * 40 * 30)
    w = 1 / np.array([10.0, 20.0, 30.0])
    assert used == "offset_weighted" and np.allclose(c, (each * w[:, None]).sum(0) / w.sum())
    c, _, used = estimate(p0, N0[:1], each[:1] - (p0 - N0[:1]), "offset_paper", eps=120.0)
    assert used == "offset" and np.allclose(c, each[0])


def test_paper_outlier_filter_pools_the_history_and_skips_the_first_frame():
    from fctrack.formation import PaperOutlierFilter
    f = PaperOutlierFilter(history=30, percentile=95.0, eps=1.0)
    rng = np.random.default_rng(1)
    base = rng.normal(0, 5, (8, 2)) + 100
    assert not f.update(base).any()
    for _ in range(40):
        flag = f.update(base + rng.normal(0, 0.5, base.shape))
    away = base.copy()
    away[3] += [200, 0]
    flag = f.update(away)
    assert flag[3] and flag.sum() == 1


def test_paper_resized_crop_is_scale_normalised():
    from fctrack.verifier import PAPER_H, PAPER_W, PR_ZOOM, paper_resized_crop
    img = np.zeros((400, 600, 3), np.uint8)
    img[195:205, 295:305] = 255  # a 10 px square at (300, 200)
    x = paper_resized_crop(img, 299.5, 199.5, 10.0)  # OpenCV puts pixel centres at integer coordinates
    assert x.shape == (3, PAPER_H, PAPER_W)
    bright = x[0] > 0
    ys, xs = np.nonzero(bright)
    assert abs(ys.mean() - PAPER_H / 2) < 1.5 and abs(xs.mean() - PAPER_W / 2) < 1.5
    assert abs(bright.sum(1).max() - PAPER_H / PR_ZOOM) < 3  # the square spans one third of the crop height
