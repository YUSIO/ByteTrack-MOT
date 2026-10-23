"""Cache YOLOX detections for offline tracker development.

Runs YOLOX inference on each frame of each sequence in a dataset,
then saves the detections to text files in MOT format. This is a
one-time GPU cost; afterwards, tracker iteration uses only CPU.

Output format (per line):
    frame_id, -1, x1, y1, w, h, confidence, -1, -1, -1

Usage:
    python tools/cache_detections.py \
        --exp-file external/YOLOX/exps/default/yolox_s.py \
        --weights weights/bytetrack_s_mot17.pth.tar \
        --data data/DanceTrack/val \
        --output detections/dancetrack_val_yolox_s/ \
        --fp16

Requires:
    - YOLOX cloned at external/YOLOX/ and on PYTHONPATH
    - Pre-trained YOLOX weights
    - GPU with sufficient VRAM (~1.5-2GB for YOLOX-S FP16)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger

# Add YOLOX to path
_YOLOX_DIR = Path(__file__).parent.parent / "external" / "YOLOX"
if str(_YOLOX_DIR) not in sys.path:
    sys.path.insert(0, str(_YOLOX_DIR))

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess

from arcsort.utils import setup_logging


# ImageNet normalization constants used by OC-SORT/ByteTrack YOLOX training
_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preproc_normalized(
    image: np.ndarray,
    input_size: tuple[int, int],
) -> tuple[np.ndarray, float]:
    """Letterbox resize with BGR->RGB conversion and ImageNet normalization.

    Matches the preprocessing used by OC-SORT/ByteTrack during YOLOX
    training. The upstream YOLOX repo's preproc() does NOT apply
    normalization, causing incorrect detections with these weights.

    Args:
        image: BGR image from cv2.imread, shape (H, W, 3).
        input_size: Target size as (height, width).

    Returns:
        Tuple of (preprocessed image in CHW float32, scale ratio).
    """
    padded = np.ones((input_size[0], input_size[1], 3), dtype=np.float32) * 114.0
    ratio = min(input_size[0] / image.shape[0], input_size[1] / image.shape[1])
    resized = cv2.resize(
        image,
        (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded[:int(image.shape[0] * ratio), :int(image.shape[1] * ratio)] = resized

    # BGR -> RGB
    padded = padded[:, :, ::-1]
    # Normalize to [0, 1] then apply ImageNet mean/std
    padded /= 255.0
    padded -= _RGB_MEAN
    padded /= _RGB_STD
    # HWC -> CHW
    padded = padded.transpose(2, 0, 1)
    padded = np.ascontiguousarray(padded, dtype=np.float32)
    return padded, ratio


class YOLOXDetector:
    """YOLOX object detector for caching detections.

    Loads a pretrained YOLOX model and runs inference on individual frames.
    Handles preprocessing (letterbox + normalize) and postprocessing
    (NMS + coordinate rescaling).
    """

    def __init__(
        self,
        exp_file: str,
        weights_path: str,
        test_size: tuple[int, int] = (608, 1088),
        conf_thresh: float = 0.01,
        nms_thresh: float = 0.7,
        fp16: bool = False,
    ) -> None:
        """Initialize detector and load model weights.

        Args:
            exp_file: Path to YOLOX experiment config file.
            weights_path: Path to pretrained model weights.
            test_size: Model input size as (height, width).
            conf_thresh: Confidence threshold for detection filtering.
            nms_thresh: NMS IoU threshold.
            fp16: Use FP16 inference for speed.

        Raises:
            FileNotFoundError: If exp_file or weights_path do not exist.
        """
        if not Path(exp_file).exists():
            raise FileNotFoundError(f"YOLOX exp file not found: {exp_file}")
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        self.test_size = test_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.fp16 = fp16

        # Load experiment config
        exp = get_exp(exp_file, None)
        exp.test_conf = conf_thresh
        exp.nmsthre = nms_thresh
        exp.test_size = test_size
        # MOT weights use num_classes=1 (person only); detect from checkpoint
        ckpt_tmp = torch.load(weights_path, map_location="cpu", weights_only=False)
        ckpt_nc = ckpt_tmp["model"]["head.cls_preds.0.weight"].shape[0]
        del ckpt_tmp
        if ckpt_nc != exp.num_classes:
            logger.info(f"Overriding num_classes {exp.num_classes} -> {ckpt_nc}")
            exp.num_classes = ckpt_nc
        self.num_classes = exp.num_classes

        # Build and load model
        model = exp.get_model()
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model = fuse_model(model)
        model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if fp16:
            model = model.half()
        model = model.to(self.device)
        self.model = model

        logger.info(
            f"YOLOX loaded: test_size={test_size}, "
            f"conf={conf_thresh}, nms={nms_thresh}, "
            f"fp16={fp16}, device={self.device}"
        )

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Run detection on a single frame.

        Args:
            frame: BGR image as numpy array, shape (H, W, 3).

        Returns:
            Detections as array of shape (N, 6) with columns
            [x1, y1, x2, y2, confidence, class_id] in original
            image coordinates.
        """
        img_h, img_w = frame.shape[:2]

        # Preprocess: letterbox resize + BGR->RGB + normalize + HWC->CHW
        # The OC-SORT/ByteTrack YOLOX models were trained with ImageNet
        # normalization applied during preprocessing. The upstream YOLOX
        # repo's preproc() does NOT normalize, so we use our own pipeline
        # matching the OC-SORT training preprocessing exactly.
        img, ratio = _preproc_normalized(frame, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()

        # Forward pass
        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs,
                self.num_classes,
                self.conf_thresh,
                self.nms_thresh,
            )

        if outputs[0] is None:
            return np.empty((0, 6))

        output = outputs[0].cpu().numpy()

        # output format: [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
        # Scale coordinates back to original image space
        bboxes = output[:, :4] / ratio
        scores = output[:, 4] * output[:, 5]
        class_ids = output[:, 6]

        # Clip to image bounds
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, img_w)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, img_h)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, img_w)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, img_h)

        return np.column_stack([bboxes, scores, class_ids])


def find_sequences(data_dir: Path) -> list[Path]:
    """Find all sequence directories in a dataset split.

    Looks for directories containing an img1/ subdirectory (MOT format).

    Args:
        data_dir: Path to dataset split (e.g., data/DanceTrack/val).

    Returns:
        Sorted list of sequence directory paths.

    Raises:
        FileNotFoundError: If data_dir does not exist or has no sequences.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    sequences = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and (d / "img1").exists()
    ])

    if not sequences:
        raise FileNotFoundError(
            f"No sequences found in {data_dir}. "
            f"Expected subdirectories with img1/ folders."
        )

    return sequences


def cache_sequence(
    detector: YOLOXDetector,
    seq_dir: Path,
    output_file: Path,
) -> int:
    """Run detection on all frames of a sequence and write results.

    Args:
        detector: Initialized YOLOX detector.
        seq_dir: Path to sequence directory (contains img1/).
        output_file: Path to write detection results.

    Returns:
        Total number of detections across all frames.
    """
    img_dir = seq_dir / "img1"
    img_files = sorted(img_dir.glob("*.jpg"))
    if not img_files:
        img_files = sorted(img_dir.glob("*.png"))

    if not img_files:
        logger.warning(f"No images found in {img_dir}")
        return 0

    lines = []
    total_dets = 0

    for frame_idx, img_file in enumerate(img_files):
        frame_id = frame_idx + 1
        frame = cv2.imread(str(img_file))
        if frame is None:
            logger.warning(f"Failed to read image: {img_file}")
            continue

        detections = detector.detect(frame)
        total_dets += detections.shape[0]

        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            w = x2 - x1
            h = y2 - y1
            lines.append(
                f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.6f},-1,-1,-1"
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")

    return total_dets


def main() -> None:
    """Main entry point for detection caching."""
    parser = argparse.ArgumentParser(
        description="Cache YOLOX detections for offline tracking"
    )
    parser.add_argument(
        "--exp-file", type=str,
        default="external/YOLOX/exps/default/yolox_s.py",
        help="Path to YOLOX experiment config",
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to pretrained YOLOX weights",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to dataset split directory (e.g., data/DanceTrack/val)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Directory to write cached detection files",
    )
    parser.add_argument(
        "--test-size", type=int, nargs=2, default=[608, 1088],
        help="Model input size as H W (default: 608 1088)",
    )
    parser.add_argument(
        "--conf-thresh", type=float, default=0.01,
        help="Detection confidence threshold (default: 0.01)",
    )
    parser.add_argument(
        "--nms-thresh", type=float, default=0.7,
        help="NMS IoU threshold (default: 0.7)",
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Use FP16 inference",
    )
    parser.add_argument(
        "--mot17-dedup", action="store_true",
        help="MOT17 deduplication: all 3 variants (DPM/FRCNN/SDP) share "
             "identical images. Cache detections only once per base sequence "
             "(e.g., MOT17-01) using the SDP variant's images.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    detector = YOLOXDetector(
        exp_file=args.exp_file,
        weights_path=args.weights,
        test_size=tuple(args.test_size),
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        fp16=args.fp16,
    )

    data_dir = Path(args.data)
    output_dir = Path(args.output)
    sequences = find_sequences(data_dir)

    logger.info(f"Found {len(sequences)} sequences in {data_dir}")

    # MOT17 dedup: pick one variant per base sequence (all share same images)
    if args.mot17_dedup:
        import re

        seen_bases: set[str] = set()
        deduped: list[Path] = []
        for seq_dir in sequences:
            # Strip variant suffix: MOT17-02-SDP -> MOT17-02
            base = re.sub(r"-(DPM|FRCNN|SDP)$", "", seq_dir.name)
            if base not in seen_bases:
                seen_bases.add(base)
                deduped.append(seq_dir)
        logger.info(
            f"MOT17 dedup: {len(sequences)} variants -> "
            f"{len(deduped)} unique sequences"
        )
        sequences = deduped

    for seq_dir in sequences:
        seq_name = seq_dir.name
        # For MOT17 dedup, save as base name (e.g., MOT17-02 not MOT17-02-SDP)
        if args.mot17_dedup:
            import re

            output_name = re.sub(r"-(DPM|FRCNN|SDP)$", "", seq_name)
        else:
            output_name = seq_name
        output_file = output_dir / f"{output_name}.txt"

        logger.info(f"Processing: {seq_name} -> {output_name}")
        total_dets = cache_sequence(detector, seq_dir, output_file)
        logger.info(f"  {total_dets} detections cached to {output_file}")

    logger.info(f"All detections cached to {output_dir}")


if __name__ == "__main__":
    main()
