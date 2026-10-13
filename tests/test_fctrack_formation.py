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
