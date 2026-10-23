"""Unified detection interface for ARC-SORT pipeline.

Wraps multiple detection backends behind a common interface.
All detectors return (N, 6) arrays with [x1, y1, x2, y2, conf, class_id].
class_id (and the ``classes`` filter) always uses the contiguous 0-indexed
COCO-80 convention of Ultralytics, so person = 0 for every backend.

Supported backends:
    - ultralytics: YOLOv8/9/10/11, RT-DETR (default)
    - yolox: YOLOX with ImageNet normalization
    - rfdetr: RF-DETR from Roboflow
    - rtdetrv2: RT-DETRv2 from HuggingFace Transformers

Usage:
    from arcsort.detection import create_detector

    detector = create_detector("yolo11x.pt", imgsz=1440, classes=[0])
    dets = detector.detect(frame)  # (N, 6) [x1,y1,x2,y2,conf,cls]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger


class UltralyticsDetector:
    """Detection via Ultralytics (YOLOv8/9/10/11, RT-DETR)."""

    def __init__(
        self,
        model_path: str,
        imgsz: int = 1440,
        conf_thresh: float = 0.01,
        classes: list[int] | None = None,
        half: bool = True,
    ) -> None:
        from ultralytics import YOLO

        # Check weights/ directory first
        weights_dir = Path(__file__).parent.parent / "weights"
        candidate = weights_dir / Path(model_path).name
        if candidate.exists():
            model_path = str(candidate)

        logger.info(f"Loading Ultralytics model: {model_path}")
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self.classes = classes
        self.half = half

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Run detection. Returns (N, 6) [x1,y1,x2,y2,conf,cls]."""
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf_thresh,
            classes=self.classes,
            verbose=False,
            half=self.half,
        )
        if len(results) == 0 or results[0].boxes is None:
            return np.empty((0, 6))

        boxes = results[0].boxes
        if boxes.shape[0] == 0:
            return np.empty((0, 6))

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()
        return np.column_stack([xyxy, confs, cls_ids])


# ImageNet normalization for YOLOX (OC-SORT/ByteTrack training)
_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _yolox_preproc(
    image: np.ndarray, input_size: tuple[int, int],
) -> tuple[np.ndarray, float]:
    """Letterbox + BGR->RGB + ImageNet normalize for YOLOX."""
    import cv2

    padded = np.ones((input_size[0], input_size[1], 3), dtype=np.float32) * 114.0
    ratio = min(input_size[0] / image.shape[0], input_size[1] / image.shape[1])
    resized = cv2.resize(
        image,
        (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded[: int(image.shape[0] * ratio), : int(image.shape[1] * ratio)] = resized
    padded = padded[:, :, ::-1]  # BGR -> RGB
    padded /= 255.0
    padded -= _RGB_MEAN
    padded /= _RGB_STD
    padded = padded.transpose(2, 0, 1)  # HWC -> CHW
    return np.ascontiguousarray(padded, dtype=np.float32), ratio


class YOLOXDetector:
    """YOLOX detector with ImageNet normalization.

    Requires external/YOLOX/ cloned and pretrained weights.
    """

    def __init__(
        self,
        exp_file: str,
        weights_path: str,
        test_size: tuple[int, int] = (608, 1088),
        conf_thresh: float = 0.01,
        nms_thresh: float = 0.7,
        fp16: bool = False,
        classes: list[int] | None = None,
    ) -> None:
        import sys

        import torch

        yolox_dir = Path(__file__).parent.parent / "external" / "YOLOX"
        if str(yolox_dir) not in sys.path:
            sys.path.insert(0, str(yolox_dir))

        try:
            from yolox.exp import get_exp
            from yolox.utils import fuse_model
        except ImportError as e:
            raise ImportError(
                "The YOLOX backend needs the YOLOX package: clone "
                "https://github.com/Megvii-BaseDetection/YOLOX and put it on "
                "PYTHONPATH (it also needs `pip install tabulate`)."
            ) from e

        self.test_size = test_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.fp16 = fp16
        self.classes = classes

        exp = get_exp(exp_file, None)
        exp.test_conf = conf_thresh
        exp.nmsthre = nms_thresh
        exp.test_size = test_size
        self.num_classes = exp.num_classes

        model = exp.get_model()
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model = fuse_model(model)
        model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if fp16:
            model = model.half()
        self.model = model.to(self.device)
        logger.info(f"YOLOX loaded: size={test_size}, fp16={fp16}")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Run detection. Returns (N, 6) [x1,y1,x2,y2,conf,cls]."""
        import torch

        img_h, img_w = frame.shape[:2]
        img, ratio = _yolox_preproc(frame, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()

        with torch.no_grad():
            from yolox.utils import postprocess

            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.conf_thresh, self.nms_thresh,
            )

        if outputs[0] is None:
            return np.empty((0, 6))

        output = outputs[0].cpu().numpy()
        bboxes = output[:, :4] / ratio
        scores = output[:, 4] * output[:, 5]
        class_ids = output[:, 6]

        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, img_w)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, img_h)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, img_w)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, img_h)

        dets = np.column_stack([bboxes, scores, class_ids])

        if self.classes is not None:
            mask = np.isin(class_ids.astype(int), self.classes)
            dets = dets[mask]

        return dets


# The 80 COCO category ids (1..90 with gaps) in order; position = contiguous index
_COCO_CATEGORY_IDS = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
     25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
     50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74,
     75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
)
_CATEGORY_TO_INDEX = np.full(_COCO_CATEGORY_IDS.max() + 1, -1, dtype=int)
_CATEGORY_TO_INDEX[_COCO_CATEGORY_IDS] = np.arange(len(_COCO_CATEGORY_IDS))


def coco_category_to_index(category_ids: np.ndarray) -> np.ndarray:
    """Map COCO category ids (person = 1) to contiguous indices (person = 0).

    Ids outside the 80 COCO categories map to -1.
    """
    cat = np.asarray(category_ids).astype(int)
    valid = (cat >= 0) & (cat < len(_CATEGORY_TO_INDEX))
    return np.where(valid, _CATEGORY_TO_INDEX[np.where(valid, cat, 0)], -1)


class RFDETRDetector:
    """RF-DETR detector from Roboflow."""

    def __init__(
        self,
        variant: str = "large",
        resolution: int = 1280,
        conf_thresh: float = 0.01,
        classes: list[int] | None = None,
    ) -> None:
        from rfdetr import RFDETRBase, RFDETRLarge

        if variant == "large":
            self.model = RFDETRLarge(resolution=resolution)
        else:
            self.model = RFDETRBase(resolution=resolution)
        self.conf_thresh = conf_thresh
        self.classes = classes
        logger.info(f"RF-DETR {variant} loaded (resolution={resolution})")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Run detection. Returns (N, 6) [x1,y1,x2,y2,conf,cls]."""
        # RF-DETR expects RGB; OpenCV frames are BGR
        results = self.model.predict(frame[:, :, ::-1].copy(), threshold=self.conf_thresh)
        if results is None or len(results.xyxy) == 0:
            return np.empty((0, 6))

        cls = coco_category_to_index(results.class_id)
        dets = np.column_stack([results.xyxy, results.confidence, cls])

        if self.classes is not None:
            dets = dets[np.isin(cls, self.classes)]

        return dets


class RTDETRv2Detector:
    """RT-DETRv2 from HuggingFace Transformers."""

    def __init__(
        self,
        model_id: str = "PekingU/rtdetr_v2_r101vd",
        conf_thresh: float = 0.01,
        classes: list[int] | None = None,
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(model_id)
        self.model.to(self.device).eval()
        self.conf_thresh = conf_thresh
        # HF RT-DETRv2 checkpoints already use contiguous 0-indexed COCO-80
        # labels (config.id2label[0] == "person")
        self.classes = classes
        logger.info(f"RT-DETRv2 loaded: {model_id} on {self.device}")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Run detection. Returns (N, 6) [x1,y1,x2,y2,conf,cls]."""
        import cv2
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        h_img, w_img = frame.shape[:2]

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([[h_img, w_img]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.conf_thresh,
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()

        if len(boxes) == 0:
            return np.empty((0, 6))

        dets = np.column_stack([boxes, scores, labels])

        if self.classes is not None:
            dets = dets[np.isin(labels, self.classes)]

        return dets


def create_detector(
    model: str,
    imgsz: int = 1440,
    conf_thresh: float = 0.01,
    classes: list[int] | None = None,
    half: bool = True,
    yolox_exp: str | None = None,
):
    """Factory: create detector from model name.

    Auto-detects backend:
        - "rfdetr-base" / "rfdetr-large" -> RF-DETR
        - "PekingU/..." -> RT-DETRv2 (HuggingFace)
        - with yolox_exp -> YOLOX
        - else -> Ultralytics (YOLOv8/9/10/11, RT-DETR)

    Args:
        model: Model name or path.
        imgsz: Detection image size (longest side).
        conf_thresh: Minimum confidence threshold.
        classes: Class IDs to keep (0=person). None for all.
        half: Use FP16 inference.
        yolox_exp: YOLOX experiment file (enables YOLOX backend).

    Returns:
        Detector instance with a .detect(frame) method.
    """
    model_lower = model.lower()

    if model_lower.startswith("rfdetr"):
        variant = "large" if "large" in model_lower else "base"
        return RFDETRDetector(
            variant=variant, resolution=imgsz,
            conf_thresh=conf_thresh, classes=classes,
        )

    if "/" in model and ("rtdetr" in model_lower or "pekingu" in model_lower):
        return RTDETRv2Detector(
            model_id=model, conf_thresh=conf_thresh, classes=classes,
        )

    if yolox_exp is not None:
        return YOLOXDetector(
            exp_file=yolox_exp, weights_path=model,
            test_size=(imgsz, imgsz),
            conf_thresh=conf_thresh, fp16=half, classes=classes,
        )

    return UltralyticsDetector(
        model_path=model, imgsz=imgsz,
        conf_thresh=conf_thresh, classes=classes, half=half,
    )
