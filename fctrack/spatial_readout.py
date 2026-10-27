"""Frozen-box regional feature readout, not an image mask or a box refinement."""
import numpy as np


def region_weights(raw, reg, grid=4, size=64):
    raw, reg = np.asarray(raw, float), np.asarray(reg, float)
    assert raw.shape == reg.shape and raw.ndim == 2 and raw.shape[1] == 4
    assert np.isfinite(raw).all() and np.isfinite(reg).all() and (raw[:, 2:] > 0).all()
    side = np.sqrt(raw[:, 2] * raw[:, 3]); center = raw[:, :2] + raw[:, 2:] / 2
    scale = size / (3 * side)
    # Same float32 affine coefficients as seed_crop.
    s = scale.astype(np.float32).astype(float)
    translation = (size / 2 - scale[:, None] * center).astype(np.float32).astype(float)
    lo = reg[:, :2] * s[:, None] + translation
    hi = (reg[:, :2] + np.maximum(reg[:, 2:], 0)) * s[:, None] + translation
    edges = np.linspace(0, size, grid + 1)
    overlap = np.maximum(0, np.minimum(hi[:, :, None], edges[None, None, 1:]) - np.maximum(lo[:, :, None], edges[None, None, :-1]))
    return overlap[:, 1, :, None] * overlap[:, 0, None, :] / (size / grid)**2


def regional_means(feature_map, weights, roll=False):
    h = np.asarray(feature_map, float); w = np.asarray(weights, float)
    assert h.ndim == 4 and w.shape == (h.shape[0], h.shape[2], h.shape[3])
    assert np.isfinite(h).all() and np.isfinite(w).all() and ((w >= 0) & (w <= 1+1e-12)).all()
    if roll:
        h = np.roll(h, (h.shape[2]//2, h.shape[3]//2), axis=(2, 3))
    def pool(mask):
        denominator = mask.sum((1, 2))[:, None]
        numerator = (h * mask[:, None]).sum((2, 3))
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return np.concatenate((pool(w), pool(1-w)), axis=1)
