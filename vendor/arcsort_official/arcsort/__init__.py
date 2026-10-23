"""ARC-SORT: Adaptive, Robust, Compensated SORT.

A multi-object tracking algorithm built on OC-SORT, designed for tracking
objects with non-linear motion under moving camera conditions.

    from arcsort import ARCTracker
    tracker = ARCTracker.from_preset("sportsmot")
    tracks = tracker.update(detections, frame)  # (N, 6) -> (M, 5)
"""

from __future__ import annotations

__version__ = "0.1.1"

from loguru import logger

# Library code stays silent unless the application opts in with
# logger.enable("arcsort") (the repo's tools do this via setup_logging).
logger.disable("arcsort")

from arcsort.tracker import ARCTracker, load_config  # noqa: E402

__all__ = ["ARCTracker", "load_config", "__version__"]
