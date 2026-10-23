"""Camera Motion Compensation via sparse optical flow.

Estimates the inter-frame camera motion as a 2D affine transform using
Lucas-Kanade sparse optical flow and RANSAC-based affine estimation.
The resulting transform is applied to Kalman filter predictions to
compensate for camera panning, tilting, and zooming.

This module is inspired by BoT-SORT's GMC (Global Motion Compensation)
but simplified to only support sparse optical flow, which provides the
best speed-accuracy trade-off for real-time tracking.
"""

from __future__ import annotations

import cv2
import numpy as np
from loguru import logger


# Default CMC parameters (overridden by config)
DEFAULT_CMC_CONFIG = {
    "max_corners": 1000,
    "quality_level": 0.01,
    "min_distance": 1,
    "block_size": 3,
    "ransac_threshold": 3.0,
    "min_inliers": 10,
    "downscale": 2,
}


class CameraMotionCompensator:
    """Estimate inter-frame camera motion using sparse optical flow.

    Tracks corner features between consecutive frames to estimate a
    2D similarity transform (rotation, uniform scale, translation).
    Falls back to identity transform when too few correspondences are
    found (e.g., uniform backgrounds, heavy motion blur).
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize CMC module.

        Args:
            config: CMC configuration dictionary. Uses DEFAULT_CMC_CONFIG
                for any missing keys.
        """
        if config is None:
            config = {}
        cfg = {**DEFAULT_CMC_CONFIG, **config}

        self.downscale = max(1, int(cfg["downscale"]))
        self.min_inliers = cfg["min_inliers"]
        self.ransac_threshold = cfg["ransac_threshold"]

        self.feature_params = {
            "maxCorners": cfg["max_corners"],
            "qualityLevel": cfg["quality_level"],
            "minDistance": cfg["min_distance"],
            "blockSize": cfg["block_size"],
            "useHarrisDetector": False,
            "k": 0.04,
        }

        self._prev_frame: np.ndarray | None = None
        self._prev_keypoints: np.ndarray | None = None

    def reset(self) -> None:
        """Reset CMC state for a new sequence."""
        self._prev_frame = None
        self._prev_keypoints = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Compute affine transform between previous and current frame.

        Args:
            frame: Current frame as BGR image, shape (H, W, 3).

        Returns:
            2x3 affine matrix representing camera motion from previous
            frame to current frame. Returns identity if this is the
            first frame or if estimation fails.
        """
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.downscale > 1:
            gray = cv2.resize(
                gray, (width // self.downscale, height // self.downscale)
            )

        keypoints = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)

        # First frame: store and return identity
        if self._prev_frame is None:
            self._prev_frame = gray
            self._prev_keypoints = keypoints
            return np.eye(2, 3, dtype=np.float64)

        affine = self._estimate_affine(gray, keypoints)

        # Store for next frame
        self._prev_frame = gray
        self._prev_keypoints = keypoints

        return affine

    def _estimate_affine(
        self, curr_frame: np.ndarray, curr_keypoints: np.ndarray | None
    ) -> np.ndarray:
        """Estimate affine transform via optical flow and RANSAC.

        Args:
            curr_frame: Current grayscale frame (possibly downscaled).
            curr_keypoints: Feature points detected in current frame
                (used for next iteration, not directly in flow).

        Returns:
            2x3 affine matrix. Identity if estimation fails.
        """
        identity = np.eye(2, 3, dtype=np.float64)

        if self._prev_keypoints is None or len(self._prev_keypoints) < 4:
            return identity

        # Track previous keypoints into current frame
        matched_pts, status, _err = cv2.calcOpticalFlowPyrLK(
            self._prev_frame, curr_frame, self._prev_keypoints, None
        )

        if matched_pts is None:
            return identity

        # Keep only successfully tracked points
        mask = status.flatten() == 1
        prev_good = self._prev_keypoints[mask]
        curr_good = matched_pts[mask]

        if len(prev_good) < self.min_inliers:
            logger.debug(
                f"CMC: too few correspondences ({len(prev_good)}), "
                f"using identity"
            )
            return identity

        # Estimate similarity transform (4 DOF: rotation, scale, tx, ty)
        affine, inliers = cv2.estimateAffinePartial2D(
            prev_good, curr_good,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
        )

        if affine is None:
            logger.debug("CMC: RANSAC failed, using identity")
            return identity

        num_inliers = int(inliers.sum()) if inliers is not None else 0
        if num_inliers < self.min_inliers:
            logger.debug(
                f"CMC: too few inliers ({num_inliers}), using identity"
            )
            return identity

        # Correct translation for downscaling
        if self.downscale > 1:
            affine[0, 2] *= self.downscale
            affine[1, 2] *= self.downscale

        return affine
