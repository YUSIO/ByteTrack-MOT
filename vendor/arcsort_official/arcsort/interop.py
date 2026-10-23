"""Conversions between ARC-SORT arrays and ``supervision.Detections``.

Requires the ``viz`` extra (``pip install arcsort[viz]``).

    import supervision as sv
    from arcsort import ARCTracker
    from arcsort.interop import from_supervision, to_supervision

    tracker = ARCTracker.from_preset("sportsmot")
    tracks = tracker.update(from_supervision(detections), frame)
    tracked = to_supervision(tracks)  # sv.Detections with tracker_id
"""

from __future__ import annotations

import numpy as np


def from_supervision(detections) -> np.ndarray:
    """``sv.Detections`` -> (N, 6) array [x1, y1, x2, y2, confidence, class_id].

    Missing confidences default to 1.0 and missing class ids to 0.
    """
    n = len(detections)
    out = np.zeros((n, 6), dtype=np.float64)
    if n == 0:
        return out
    out[:, :4] = detections.xyxy
    out[:, 4] = 1.0 if detections.confidence is None else detections.confidence
    out[:, 5] = 0 if detections.class_id is None else detections.class_id
    return out


def to_supervision(tracks: np.ndarray):
    """(M, 5) tracker output [x1, y1, x2, y2, track_id] -> ``sv.Detections``."""
    import supervision as sv

    tracks = np.asarray(tracks, dtype=np.float64).reshape(-1, 5)
    return sv.Detections(
        xyxy=tracks[:, :4].astype(np.float32),
        tracker_id=tracks[:, 4].astype(int),
    )
