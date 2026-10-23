"""Bounding box format conversions and shared utilities.

Provides conversion functions between bbox formats used throughout the
ARC-SORT pipeline. All bounding boxes are stored as numpy arrays.

Format conventions:
    xyxy:   [x1, y1, x2, y2] -- top-left and bottom-right corners
    xywh:   [x, y, w, h]     -- top-left corner, width, height
    cxcywh: [cx, cy, w, h]   -- center, width, height
    cxcysr: [cx, cy, s, r]   -- center, scale (area), aspect ratio (w/h)
"""

from __future__ import annotations

import sys

import numpy as np
from loguru import logger


def setup_logging(verbose: bool = False) -> None:
    """Configure loguru logging for the ARC-SORT pipeline.

    Args:
        verbose: If True, set level to DEBUG. Otherwise INFO.
    """
    logger.enable("arcsort")
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level, format="{time:HH:mm:ss} | {level:<7} | {message}")


def xyxy_to_xywh(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [x, y, w, h].

    Args:
        bbox: Bounding box(es) in xyxy format. Shape (..., 4).

    Returns:
        Bounding box(es) in xywh format. Same shape as input.
    """
    result = bbox.copy()
    result[..., 2] = bbox[..., 2] - bbox[..., 0]
    result[..., 3] = bbox[..., 3] - bbox[..., 1]
    return result


def xywh_to_xyxy(bbox: np.ndarray) -> np.ndarray:
    """Convert [x, y, w, h] to [x1, y1, x2, y2].

    Args:
        bbox: Bounding box(es) in xywh format. Shape (..., 4).

    Returns:
        Bounding box(es) in xyxy format. Same shape as input.
    """
    result = bbox.copy()
    result[..., 2] = bbox[..., 0] + bbox[..., 2]
    result[..., 3] = bbox[..., 1] + bbox[..., 3]
    return result


def xyxy_to_cxcywh(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [cx, cy, w, h].

    Args:
        bbox: Bounding box(es) in xyxy format. Shape (..., 4).

    Returns:
        Bounding box(es) in cxcywh format. Same shape as input.
    """
    result = np.empty_like(bbox)
    w = bbox[..., 2] - bbox[..., 0]
    h = bbox[..., 3] - bbox[..., 1]
    result[..., 0] = bbox[..., 0] + w / 2
    result[..., 1] = bbox[..., 1] + h / 2
    result[..., 2] = w
    result[..., 3] = h
    return result


def cxcywh_to_xyxy(bbox: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2].

    Args:
        bbox: Bounding box(es) in cxcywh format. Shape (..., 4).

    Returns:
        Bounding box(es) in xyxy format. Same shape as input.
    """
    result = np.empty_like(bbox)
    half_w = bbox[..., 2] / 2
    half_h = bbox[..., 3] / 2
    result[..., 0] = bbox[..., 0] - half_w
    result[..., 1] = bbox[..., 1] - half_h
    result[..., 2] = bbox[..., 0] + half_w
    result[..., 3] = bbox[..., 1] + half_h
    return result


def xyxy_to_cxcysr(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [cx, cy, s, r].

    The Kalman filter operates in this space where s = area (w*h)
    and r = aspect ratio (w/h).

    Args:
        bbox: Bounding box(es) in xyxy format. Shape (..., 4).

    Returns:
        Bounding box(es) in cxcysr format. Same shape as input.
    """
    result = np.empty_like(bbox)
    w = bbox[..., 2] - bbox[..., 0]
    h = bbox[..., 3] - bbox[..., 1]
    result[..., 0] = bbox[..., 0] + w / 2
    result[..., 1] = bbox[..., 1] + h / 2
    result[..., 2] = w * h
    result[..., 3] = w / (h + 1e-6)
    return result


def cxcysr_to_xyxy(bbox: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, s, r] to [x1, y1, x2, y2].

    Args:
        bbox: Bounding box(es) in cxcysr format. Shape (..., 4).

    Returns:
        Bounding box(es) in xyxy format. Same shape as input.
    """
    result = np.empty_like(bbox)
    s = bbox[..., 2]
    r = bbox[..., 3]
    w = np.sqrt(np.maximum(s * r, 0))
    h = s / (w + 1e-6)
    result[..., 0] = bbox[..., 0] - w / 2
    result[..., 1] = bbox[..., 1] - h / 2
    result[..., 2] = bbox[..., 0] + w / 2
    result[..., 3] = bbox[..., 1] + h / 2
    return result
