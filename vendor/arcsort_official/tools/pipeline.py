"""End-to-end video pipeline for ARC-SORT.

Supports detection, tracking, and visualization in a single command.
Two processing modes:
    - online: detect + track per frame (real-time capable)
    - batch:  detect all frames first, then track (benchmark-friendly)

Supports all ARC-SORT tracker flags, multiple detector backends, pre-cached
detections, side-by-side visualization, MOT result export, and FPS overlay.

Usage:
    # Online: real-time from webcam
    PYTHONPATH=. python tools/pipeline.py --source 0 \
        --detector yolo11m.pt \
        --config arcsort/configs/sportsmot.yaml \
        --mode online --show

    # Batch: process video file with full pipeline
    PYTHONPATH=. python tools/pipeline.py --source match.mp4 \
        --detector yolo11x.pt \
        --config arcsort/configs/sportsmot.yaml \
        --mode batch --save output.mp4 --show

    # From pre-cached detections (skip detection)
    PYTHONPATH=. python tools/pipeline.py --source video.mp4 \
        --detections cached_dets.txt \
        --config arcsort/configs/sportsmot.yaml \
        --save output.mp4

    # Side-by-side: original + annotated
    PYTHONPATH=. python tools/pipeline.py --source match.mp4 \
        --detector yolo11x.pt \
        --config arcsort/configs/sportsmot.yaml \
        --mode online --show --side-by-side

    # Save tracking results in MOT format for evaluation
    PYTHONPATH=. python tools/pipeline.py --source match.mp4 \
        --detector yolo11x.pt \
        --config arcsort/configs/sportsmot.yaml \
        --save output.mp4 --save-mot results.txt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from arcsort.detection import create_detector
from arcsort.tracker import ARCTracker, load_config
from arcsort.utils import setup_logging
from arcsort.visualizer import Visualizer


# ---------------------------------------------------------------------------
# Video source abstraction
# ---------------------------------------------------------------------------

class VideoSource:
    """Unified video source: file, webcam, or frame directory."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.is_webcam = source.isdigit()
        self.is_directory = Path(source).is_dir()
        self._cap: cv2.VideoCapture | None = None
        self._frame_files: list[Path] | None = None
        self._frame_idx = 0

        if self.is_webcam:
            self._cap = cv2.VideoCapture(int(source))
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open webcam {source}")
        elif self.is_directory:
            exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            files: list[Path] = []
            for ext in exts:
                files.extend(Path(source).glob(ext))
            self._frame_files = sorted(files)
            if not self._frame_files:
                raise FileNotFoundError(f"No images found in {source}")
        else:
            self._cap = cv2.VideoCapture(source)
            if not self._cap.isOpened():
                raise FileNotFoundError(f"Cannot open video: {source}")

    @property
    def total_frames(self) -> int:
        if self.is_webcam:
            return 0
        if self.is_directory:
            return len(self._frame_files)
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self) -> float:
        if self.is_webcam or self.is_directory:
            return 30.0
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def size(self) -> tuple[int, int]:
        """Returns (width, height)."""
        if self.is_directory:
            frame = cv2.imread(str(self._frame_files[0]))
            return frame.shape[1], frame.shape[0]
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read next frame."""
        if self.is_directory:
            if self._frame_idx >= len(self._frame_files):
                return False, None
            frame = cv2.imread(str(self._frame_files[self._frame_idx]))
            self._frame_idx += 1
            return frame is not None, frame
        return self._cap.read()

    def reset(self) -> None:
        """Reset to first frame (for batch mode second pass)."""
        if self.is_directory:
            self._frame_idx = 0
        elif self._cap:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self) -> None:
        """Release video capture resources."""
        if self._cap:
            self._cap.release()


# ---------------------------------------------------------------------------
# MOT format I/O
# ---------------------------------------------------------------------------

def load_cached_detections(
    det_file: str,
    min_confidence: float = 0.0,
) -> dict[int, np.ndarray]:
    """Load pre-cached detections from MOT format file.

    Args:
        det_file: Path to detection file with format:
            frame_id, -1, x1, y1, w, h, confidence, -1, -1, -1
        min_confidence: Discard detections below this threshold.

    Returns:
        Dict mapping frame_id to (N, 6) [x1,y1,x2,y2,conf,cls].
    """
    path = Path(det_file)
    if not path.exists():
        raise FileNotFoundError(f"Detection file not found: {det_file}")

    data = np.loadtxt(str(path), delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] == 0:
        return {}

    if min_confidence > 0:
        data = data[data[:, 6] >= min_confidence]
        if data.shape[0] == 0:
            return {}

    result: dict[int, np.ndarray] = {}
    frame_ids = data[:, 0].astype(int)
    for fid in np.unique(frame_ids):
        fd = data[frame_ids == fid]
        x1, y1 = fd[:, 2], fd[:, 3]
        w, h = fd[:, 4], fd[:, 5]
        conf = fd[:, 6]
        result[fid] = np.column_stack([
            x1, y1, x1 + w, y1 + h, conf, np.zeros(len(conf)),
        ])
    return result


def save_mot_results(
    results: dict[int, np.ndarray],
    output_path: str,
) -> None:
    """Save tracking results in MOT format.

    Args:
        results: Dict mapping frame_id to (M, 5) [x1,y1,x2,y2,tid].
        output_path: Path to write output file.
    """
    lines = []
    for frame_id in sorted(results.keys()):
        tracks = results[frame_id]
        for row in tracks:
            x1, y1, x2, y2, tid = row
            w, h = x2 - x1, y2 - y1
            lines.append(
                f"{frame_id},{int(tid)},{x1:.2f},{y1:.2f},"
                f"{w:.2f},{h:.2f},1,-1,-1,-1"
            )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")
    logger.info(f"MOT results saved to {output_path}")


# ---------------------------------------------------------------------------
# Video writer helper
# ---------------------------------------------------------------------------

def _create_writer(
    path: str, fps: float, width: int, height: int,
) -> cv2.VideoWriter:
    """Create video writer with codec matched to file extension."""
    ext = Path(path).suffix.lower()
    if ext == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer for {path}")
    return writer


# ---------------------------------------------------------------------------
# Processing modes
# ---------------------------------------------------------------------------

def run_online(
    source: VideoSource,
    detector,
    tracker: ARCTracker,
    visualizer: Visualizer,
    args: argparse.Namespace,
) -> None:
    """Online mode: detect + track per frame (real-time capable)."""
    w, h = source.size
    out_w = w * 2 if args.side_by_side else w

    writer = None
    if args.save:
        writer = _create_writer(args.save, source.fps, out_w, h)

    mot_results: dict[int, np.ndarray] | None = {} if args.save_mot else None
    frame_id = 0
    fps_smooth = 0.0

    logger.info("Running online mode (press 'q' to quit)")

    while True:
        ret, frame = source.read()
        if not ret:
            break

        frame_id += 1
        t0 = time.perf_counter()

        # Detect + Track
        dets = detector.detect(frame)
        tracks = tracker.update(dets, frame=frame)

        dt = time.perf_counter() - t0
        fps = 1.0 / dt if dt > 0 else 0
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps if fps_smooth > 0 else fps

        # Visualize
        annotated = visualizer.annotate(
            frame, tracks, fps=fps_smooth, camera_motion=tracker.camera_motion,
        )

        if args.side_by_side:
            output = Visualizer.side_by_side(frame, annotated)
        else:
            output = annotated

        if args.show:
            cv2.imshow("ARC-SORT Pipeline", output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Quit by user")
                break

        if writer:
            writer.write(output)

        if mot_results is not None:
            mot_results[frame_id] = tracks.copy()

    if writer:
        writer.release()
        logger.info(f"Video saved to {args.save}")
    if mot_results:
        save_mot_results(mot_results, args.save_mot)
    if args.show:
        cv2.destroyAllWindows()


def run_batch(
    source: VideoSource,
    detector,
    tracker: ARCTracker,
    visualizer: Visualizer,
    args: argparse.Namespace,
    cached_dets: dict[int, np.ndarray] | None = None,
) -> None:
    """Batch mode: detect all frames, then track all frames.

    If cached_dets is provided, the detection pass is skipped.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    total = source.total_frames

    # Pass 1: Detection
    if cached_dets is None:
        logger.info(f"Pass 1/2: Detecting on {total} frames...")
        cached_dets = {}
        iterator = range(total)
        if tqdm:
            iterator = tqdm(iterator, desc="Detecting", unit="frame")

        for i in iterator:
            ret, frame = source.read()
            if not ret:
                break
            frame_id = i + 1
            dets = detector.detect(frame)
            if dets.shape[0] > 0:
                cached_dets[frame_id] = dets

        det_count = sum(len(d) for d in cached_dets.values())
        logger.info(f"Detection complete: {det_count} detections across {len(cached_dets)} frames")
        source.reset()
    else:
        logger.info("Using pre-cached detections, skipping detection pass")

    # Pass 2: Tracking + Visualization
    w, h = source.size
    out_w = w * 2 if args.side_by_side else w
    max_frame = max(cached_dets.keys()) if cached_dets else total

    logger.info(f"Pass 2/2: Tracking + visualization on {max_frame} frames...")

    writer = None
    if args.save:
        writer = _create_writer(args.save, source.fps, out_w, h)

    mot_results: dict[int, np.ndarray] | None = {} if args.save_mot else None

    iterator = range(max_frame)
    if tqdm:
        iterator = tqdm(iterator, desc="Tracking", unit="frame")

    for i in iterator:
        ret, frame = source.read()
        if not ret:
            break

        frame_id = i + 1
        dets = cached_dets.get(frame_id, np.empty((0, 6)))
        tracks = tracker.update(dets, frame=frame)

        annotated = visualizer.annotate(frame, tracks, camera_motion=tracker.camera_motion)

        if args.side_by_side:
            output = Visualizer.side_by_side(frame, annotated)
        else:
            output = annotated

        if args.show:
            cv2.imshow("ARC-SORT Pipeline", output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Quit by user")
                break

        if writer:
            writer.write(output)

        if mot_results is not None:
            mot_results[frame_id] = tracks.copy()

    if writer:
        writer.release()
        logger.info(f"Video saved to {args.save}")
    if mot_results:
        save_mot_results(mot_results, args.save_mot)
    if args.show:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ARC-SORT end-to-end video pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Source
    parser.add_argument(
        "--source", type=str, required=True,
        help="Video file, webcam ID (e.g. 0), or frame directory",
    )

    # Detection
    det_group = parser.add_argument_group("detection")
    det_group.add_argument(
        "--detector", type=str, default=None,
        help="Model name/path (e.g. yolo11x.pt, rfdetr-large). "
             "Required unless --detections is provided.",
    )
    det_group.add_argument(
        "--detections", type=str, default=None,
        help="Pre-cached detection file (MOT format). Skips detection.",
    )
    det_group.add_argument(
        "--imgsz", type=int, default=640,
        help="Detection image size (default: 640)",
    )
    det_group.add_argument(
        "--conf-thresh", type=float, default=0.01,
        help="Detection confidence threshold (default: 0.01)",
    )
    det_group.add_argument(
        "--classes", type=int, nargs="+", default=[0],
        help="Class IDs to track (default: 0 = person). Use -1 for all.",
    )
    det_group.add_argument(
        "--yolox-exp", type=str, default=None,
        help="YOLOX experiment file (enables YOLOX backend)",
    )
    det_group.add_argument(
        "--no-half", action="store_true",
        help="Disable FP16 inference (default: FP16 enabled)",
    )

    # Tracker
    parser.add_argument(
        "--config", type=str, required=True,
        help="Tracker config YAML file",
    )

    # Mode
    parser.add_argument(
        "--mode", type=str, default="online", choices=["online", "batch"],
        help="Processing mode (default: online)",
    )

    # Output
    out_group = parser.add_argument_group("output")
    out_group.add_argument(
        "--show", action="store_true",
        help="Display live visualization window",
    )
    out_group.add_argument(
        "--save", type=str, default=None,
        help="Save output video to path (e.g. output.mp4)",
    )
    out_group.add_argument(
        "--save-mot", type=str, default=None,
        help="Save tracking results in MOT format",
    )
    out_group.add_argument(
        "--side-by-side", action="store_true",
        help="Show original and annotated frames side by side",
    )

    # Visualization
    viz_group = parser.add_argument_group("visualization")
    viz_group.add_argument(
        "--trace-length", type=int, default=30,
        help="Track trail length in frames (default: 30)",
    )
    viz_group.add_argument(
        "--no-trace", action="store_true",
        help="Disable track trails",
    )
    viz_group.add_argument(
        "--motion-trails", action="store_true",
        help="Keep trails fixed to the scene while the camera moves, using the "
             "tracker's camera-motion estimate (needs CMC on and arcsort[viz])",
    )
    viz_group.add_argument(
        "--no-labels", action="store_true",
        help="Disable track ID labels",
    )

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    # --- Validate arguments ---
    if args.detections is None and args.detector is None:
        parser.error("--detector is required unless --detections is provided")

    if args.source.isdigit() and args.mode == "batch":
        parser.error("Batch mode is not supported with webcam source")

    if not args.show and not args.save and not args.save_mot:
        parser.error("Specify at least one output: --show, --save, or --save-mot")

    classes = args.classes if args.classes != [-1] else None
    half = not args.no_half

    # --- Load tracker ---
    config = load_config(args.config)
    tracker = ARCTracker(config)
    logger.info(f"Tracker: {config.get('name', 'ARC-SORT')}")

    # --- Open source ---
    source = VideoSource(args.source)
    info = f"Source: {args.source} | {source.size[0]}x{source.size[1]} @ {source.fps:.1f}fps"
    if source.total_frames > 0:
        info += f" | {source.total_frames} frames"
    logger.info(info)

    # --- Create detector or load cached detections ---
    detector = None
    cached_dets = None

    if args.detections:
        min_conf = config.get("track_low_thresh", 0.1)
        cached_dets = load_cached_detections(args.detections, min_confidence=min_conf)
        det_count = sum(len(d) for d in cached_dets.values())
        logger.info(f"Loaded {det_count} cached detections from {args.detections}")
    else:
        detector = create_detector(
            model=args.detector,
            imgsz=args.imgsz,
            conf_thresh=args.conf_thresh,
            classes=classes,
            half=half,
            yolox_exp=args.yolox_exp,
        )

    # --- Create visualizer ---
    trace_length = 0 if args.no_trace else args.trace_length
    visualizer = Visualizer(
        trace_length=trace_length,
        show_labels=not args.no_labels,
        motion_compensated_trails=args.motion_trails,
    )

    # --- Run pipeline ---
    try:
        if args.mode == "batch" or cached_dets is not None:
            run_batch(source, detector, tracker, visualizer, args, cached_dets)
        else:
            run_online(source, detector, tracker, visualizer, args)
    finally:
        source.release()

    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
