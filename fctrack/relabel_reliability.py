"""Exp170's frozen spread feature, computed causally for Exp171 veto only."""
import numpy as np
from fctrack.formation import select_neighbours

def saved_centre(box):
    # Match run.py's saved-output precision used to calibrate Exp170.
    b = np.array([round(v, 1) for v in box])
    return b[:2] + b[2:] / 2

def form_spread(label, c1, side1, reference, current):
    ref = dict(reference)
    ref[label] = c1
    ids = list(ref)
    points = np.array([ref[j] for j in ids])
    available = np.array([j != label and j in current for j in ids])
    nb = select_neighbours(points, ids.index(label), available, 'delaunay', 6)
    if len(nb) < 2:
        return None, len(nb)
    q = np.array([current[ids[j]] + c1 - points[j] for j in nb])
    diameter = np.linalg.norm(q[:, None] - q[None, :], axis=2).max()
    w = np.ones(len(nb)) if diameter <= .5 * side1 else 1 / np.maximum(np.linalg.norm(points[nb] - c1, axis=1), 1e-6)
    w /= w.sum()
    mu = (w[:, None] * q).sum(0)
    return float(np.sqrt((w * np.square(q - mu).sum(1)).sum()) / side1), len(nb)
