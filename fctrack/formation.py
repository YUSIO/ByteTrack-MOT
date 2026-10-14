"""Formation constraints of FCTrack (V-USDT, Rahman and Madria, MDM 2025) and graph variants of them.

A missing member is estimated from the members that were observed both in the reference frame (the last frame the
missing member was observed in) and in the current frame. Two choices are separated:

  graph   which members are used        mst       neighbours on the Euclidean minimum spanning tree (the paper)
                                        knn       nearest members
                                        delaunay  natural neighbours (Delaunay triangulation)
  model   how they move the member      offset      each neighbour keeps its offset to the member; neighbour estimates
                                                    within eps pixels of each other are averaged, otherwise weighted
                                        offset_paper  the same offsets with the consistency rule as written in the paper:
                                                    two estimates agree when their squared distance is at most a fraction
                                                    (10 %) of the box area; the agreeing estimates are used when at least
                                                    two agree, otherwise all are weighted by distance
                                        similarity  one rotation + scale + translation fitted to the neighbours
                                        affine      one affine map fitted to the neighbours; with three neighbours
                                                    this is the barycentric coordinate of the member in their triangle

The consistency rule of `offset` is not the paper's (it was written before the paper was re-read); `offset_paper` is.
"""
from collections import deque

import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import QhullError


def mst_adjacency(P):
    """Prim's algorithm on the complete Euclidean graph. Returns the adjacency lists of the tree."""
    n = len(P)
    adj = [[] for _ in range(n)]
    if n < 2:
        return adj
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    in_tree = np.zeros(n, bool)
    in_tree[0] = True
    best, parent = D[0].copy(), np.zeros(n, int)
    for _ in range(n - 1):
        j = int(np.where(in_tree, np.inf, best).argmin())
        adj[j].append(int(parent[j]))
        adj[int(parent[j])].append(j)
        in_tree[j] = True
        closer = (D[j] < best) & ~in_tree
        parent[closer], best[closer] = j, D[j][closer]
    return adj


def delaunay_adjacency(P):
    """Adjacency of the Delaunay triangulation; None when it does not exist (fewer than 3 points, or collinear)."""
    if len(P) < 3:
        return None
    try:
        tri = Delaunay(P)
    except QhullError:
        return None
    ptr, idx = tri.vertex_neighbor_vertices
    return [list(map(int, idx[ptr[i]:ptr[i + 1]])) for i in range(len(P))]


def select_neighbours(P0, m, available, graph, max_neighbours):
    """Indices of up to `max_neighbours` available members, walking outwards from member `m` on the chosen graph."""
    d = np.linalg.norm(P0 - P0[m], axis=1)
    if graph == "knn":
        order = [int(i) for i in np.argsort(d) if i != m and available[i]]
        return order[:max_neighbours]
    adj = mst_adjacency(P0) if graph == "mst" else delaunay_adjacency(P0)
    if adj is None:  # no triangulation: fall back to the nearest members
        return select_neighbours(P0, m, available, "knn", max_neighbours)
    seen, ring, out = {m}, [m], []
    while ring and len(out) < max_neighbours:  # breadth first: first ring, then the ring behind an unavailable member
        nxt = sorted({j for i in ring for j in adj[i] if j not in seen}, key=lambda j: d[j])
        seen.update(nxt)
        out += [j for j in nxt if available[j]]
        ring = nxt
    return out[:max_neighbours]


def fit_similarity(A, B):
    """Least-squares rotation + uniform scale + translation mapping points A to B (Umeyama, 2-D, no reflection)."""
    ma, mb = A.mean(0), B.mean(0)
    a, b = A - ma, B - mb
    var = (a ** 2).sum()
    if var < 1e-9:
        return np.eye(2), mb - ma
    U, S, Vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, d]) @ Vt
    s = (S * [1.0, d]).sum() / var
    return s * R, mb - s * R @ ma


def estimate(p0, N0, N1, model, eps):
    """Centre of the missing member in the current frame.

    p0: its centre in the reference frame; N0, N1: k x 2 centres of the chosen neighbours in the reference and the
    current frame; eps: neighbour estimates within `eps` pixels of each other count as consistent (offset model), or
    the largest squared distance in pixels^2 of two agreeing estimates (offset_paper).
    Returns (centre, spread, model actually used); spread is the largest distance between two neighbour estimates.
    """
    each = N1 + (p0 - N0)  # what every neighbour says on its own
    spread = float(max(np.linalg.norm(each[:, None] - each[None], axis=2).max(), 0.0)) if len(each) > 1 else 0.0
    k = len(N0)
    if model == "offset_paper" and k >= 2:
        agree = ((each[:, None] - each[None]) ** 2).sum(2) <= eps
        i = int(agree.sum(1).argmax())  # the estimate with the most agreeing others (the nearest neighbour's on ties)
        if agree[i].sum() >= 2:
            return each[agree[i]].mean(0), spread, "offset_agree"
        w = 1.0 / np.maximum(np.linalg.norm(N0 - p0, axis=1), 1e-6)
        return (each * w[:, None]).sum(0) / w.sum(), spread, "offset_weighted"
    if model == "offset_paper":
        model = "offset"
    if model == "affine" and k >= 3:
        X = np.c_[N0, np.ones(k)]
        if np.linalg.cond(X) < 1e4:  # not collinear
            lever = float(np.r_[p0, 1.0] @ np.linalg.pinv(X.T @ X) @ np.r_[p0, 1.0])
            if lever <= 3.0:  # the member is not far outside its neighbours: no wild extrapolation
                W = np.linalg.lstsq(X, N1, rcond=None)[0]
                return np.r_[p0, 1.0] @ W, spread, "affine"
        model = "similarity"
    if model in ("affine", "similarity") and k >= 2:
        M, t = fit_similarity(N0, N1)
        return M @ p0 + t, spread, "similarity"
    if k == 1 or spread <= eps:
        return each.mean(0), spread, "offset"
    w = 1.0 / np.maximum(np.linalg.norm(N0 - p0, axis=1), 1e-6)  # inconsistent: nearer neighbours count more
    return (each * w[:, None]).sum(0) / w.sum(), spread, "offset_weighted"


class PaperOutlierFilter:
    """Stage 1 of FCTrack as in Algorithm 1 of the paper.

    The centres of the last `history` frames (including the current one) are pooled; the member distances are measured
    to their mean with the covariance + eps I; the threshold is the `percentile` of the distances of the last `history`
    frames including the current one. Nothing is flagged in the first frame.
    """

    def __init__(self, history=30, percentile=95.0, eps=1.0):
        self.pc, self.dh = deque(maxlen=history), deque(maxlen=history)
        self.percentile, self.eps, self.first = percentile, eps, True

    def update(self, centres):
        centres = np.asarray(centres, float).reshape(-1, 2)
        self.pc.append(centres)
        if self.first:
            self.first = False
            return np.zeros(len(centres), bool)
        flat = np.concatenate(self.pc)
        if len(flat) > 1:
            mu, inv = flat.mean(0), np.linalg.inv(np.cov(flat.T) + self.eps * np.eye(2))
        else:
            mu, inv = np.zeros(2), np.eye(2)
        d = np.sqrt(np.einsum("ij,jk,ik->i", centres - mu, inv, centres - mu))
        self.dh.append(d)
        hist = np.concatenate(self.dh)
        return d > np.percentile(hist, self.percentile) if len(hist) else np.zeros(len(centres), bool)


class OutlierFilter:
    """Stage 1 of FCTrack: members far from the formation by Mahalanobis distance are held back.

    The distance of every member to the regularised mean / covariance of the members of its frame is kept for
    `history` frames; a member is an outlier when its distance exceeds the `percentile` of that history.
    """

    def __init__(self, history=30, percentile=95.0, regulariser=1e-2, min_members=4, min_samples=30):
        self.past = deque(maxlen=history)
        self.percentile, self.reg, self.min_members, self.min_samples = percentile, regulariser, min_members, min_samples

    def update(self, centres):
        """centres: n x 2. Returns a boolean mask of outliers and appends the frame to the history."""
        n = len(centres)
        flag = np.zeros(n, bool)
        if n < self.min_members:
            return flag
        mu = centres.mean(0)
        cov = np.cov((centres - mu).T) + self.reg * np.trace(np.cov((centres - mu).T)) / 2 * np.eye(2) + 1e-6 * np.eye(2)
        d = np.sqrt(np.einsum("ij,jk,ik->i", centres - mu, np.linalg.inv(cov), centres - mu))
        hist = np.concatenate(self.past) if self.past else np.zeros(0)
        if len(hist) >= self.min_samples:
            flag = d > np.percentile(hist, self.percentile)
        self.past.append(d)
        return flag
