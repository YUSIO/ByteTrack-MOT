"""Visualization engine using supervision for ARC-SORT pipeline.

Draws bounding boxes, track IDs, and movement trails on video frames.
Supports side-by-side view (original | annotated) and FPS overlay.

Usage:
    from arcsort.visualizer import Visualizer

    viz = Visualizer(trace_length=30)
    annotated = viz.annotate(frame, tracks, fps=25.3)
    combined = Visualizer.side_by_side(frame, annotated)

With a moving camera, plain trails also record the camera's motion. Pass
``motion_compensated_trails=True`` and the tracker's per-frame camera motion
to keep trails fixed to the scene (uses Roboflow ``trackers``'
MotionAwareTraceAnnotator):

    viz = Visualizer(motion_compensated_trails=True)
    tracks = tracker.update(dets, frame)
    annotated = viz.annotate(frame, tracks, camera_motion=tracker.camera_motion)
"""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv


class Visualizer:
    """Annotate frames with tracking results using supervision."""

    def __init__(
        self,
        trace_length: int = 30,
        show_labels: bool = True,
        motion_compensated_trails: bool = False,
    ) -> None:
        """Initialize annotators.

        Args:
            trace_length: Track trail length in frames. 0 disables trails.
            show_labels: Show track ID labels on boxes.
            motion_compensated_trails: Store trail points in scene coordinates
                and redraw them in the current view, so camera pans do not
                bend the trails. Needs ``pip install arcsort[viz]`` (Roboflow
                ``trackers``) and ``camera_motion`` passed to ``annotate()``.
        """
        self.show_labels = show_labels
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            color_lookup=sv.ColorLookup.TRACK,
        )
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_position=sv.Position.TOP_LEFT,
            color_lookup=sv.ColorLookup.TRACK,
        )
        self.trace_length = trace_length
        self.motion_compensated_trails = motion_compensated_trails and trace_length > 0
        if self.motion_compensated_trails:
            try:
                import trackers  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "motion_compensated_trails needs Roboflow trackers: pip install arcsort[viz]"
                ) from e
        self.reset()

    def reset(self) -> None:
        """Clear trails and accumulated camera motion for a new video."""
        # First frame -> current frame homography, accumulated from the
        # per-frame camera motion
        self._world_to_frame = np.eye(3)
        self.trace_annotator = None
        if self.trace_length <= 0:
            return
        kwargs = dict(
            trace_length=self.trace_length,
            thickness=2,
            position=sv.Position.BOTTOM_CENTER,
            color_lookup=sv.ColorLookup.TRACK,
        )
        if self.motion_compensated_trails:
            from trackers import MotionAwareTraceAnnotator

            self.trace_annotator = MotionAwareTraceAnnotator(**kwargs)
        else:
            self.trace_annotator = sv.TraceAnnotator(**kwargs)

    def annotate(
        self,
        frame: np.ndarray,
        tracks: np.ndarray,
        fps: float | None = None,
        camera_motion: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw tracks on frame.

        Args:
            frame: BGR image (H, W, 3).
            tracks: (M, 5) [x1, y1, x2, y2, track_id] from tracker.update().
            fps: Optional FPS to overlay on frame.
            camera_motion: 2x3 affine from the previous frame to this one,
                e.g. ``tracker.camera_motion``. Used only with
                ``motion_compensated_trails``; call once per frame, in order.

        Returns:
            Annotated copy of the frame.
        """
        annotated = frame.copy()

        if self.motion_compensated_trails and camera_motion is not None:
            step = np.vstack([np.asarray(camera_motion, dtype=np.float64), [0.0, 0.0, 1.0]])
            self._world_to_frame = step @ self._world_to_frame

        if tracks.shape[0] > 0:
            tracker_ids = tracks[:, 4].astype(int)
            detections = sv.Detections(
                xyxy=tracks[:, :4],
                tracker_id=tracker_ids,
            )

            if self.motion_compensated_trails:
                from trackers import HomographyTransformation

                annotated = self.trace_annotator.annotate(
                    annotated, detections,
                    coord_transform=HomographyTransformation(self._world_to_frame),
                )
            elif self.trace_annotator is not None:
                annotated = self.trace_annotator.annotate(annotated, detections)
            annotated = self.box_annotator.annotate(annotated, detections)

            if self.show_labels:
                labels = [f"ID:{tid}" for tid in tracker_ids]
                annotated = self.label_annotator.annotate(
                    annotated, detections, labels=labels,
                )

        if fps is not None:
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return annotated

    @staticmethod
    def side_by_side(
        original: np.ndarray, annotated: np.ndarray,
    ) -> np.ndarray:
        """Combine original and annotated frames horizontally."""
        return np.hstack([original, annotated])
