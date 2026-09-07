#!/usr/bin/env python3
"""Frozen YOLO11s P3 ROI feature-cache utilities for UAVSwarm MOT.

These helpers never read ground truth.  A feature cache is tied to one scored
detector-cache file by exact frame, box, score, and SHA-256 checks so that the
tracker cannot silently consume features from a different detector input.
"""

from __future__ import annotations

import configparser
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sequence_info(sequence_dir: Path) -> dict[str, Any]:
    parser = configparser.ConfigParser()
    parser.read(sequence_dir / "seqinfo.ini")
    section = parser["Sequence"]
    return {
        "name": section["name"],
        "frames": int(section["seqLength"]),
        "height": int(section["imHeight"]),
        "width": int(section["imWidth"]),
        "frame_rate": int(section["frameRate"]),
        "image_dir": section["imDir"],
        "image_extension": section["imExt"],
    }


def load_detections(path: Path) -> dict[int, np.ndarray]:
    """Read a scored MOT detector cache as per-frame x1,y1,x2,y2,score rows."""
    detections: dict[int, list[list[float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for line_number, row in enumerate(csv.reader(source), start=1):
            if not row:
                continue
            if len(row) < 7:
                raise ValueError(f"{path}:{line_number}: expected at least seven MOT fields")
            frame = int(float(row[0]))
            x, y, width, height, score = (float(value) for value in row[2:7])
            if not np.isfinite([x, y, width, height, score]).all():
                raise ValueError(f"{path}:{line_number}: non-finite detection value")
            if width <= 0.0 or height <= 0.0:
                raise ValueError(f"{path}:{line_number}: non-positive detection geometry")
            detections[frame].append([x, y, x + width, y + height, score])
    return {
        frame: np.asarray(rows, dtype=np.float32)
        for frame, rows in detections.items()
    }


def flatten_detections(info: dict[str, Any], detections: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_ids, boxes, scores = [], [], []
    for frame in range(1, int(info["frames"]) + 1):
        rows = detections.get(frame)
        if rows is None or len(rows) == 0:
            continue
        frame_ids.append(np.full(len(rows), frame, dtype=np.int32))
        boxes.append(rows[:, :4].astype(np.float32, copy=False))
        scores.append(rows[:, 4].astype(np.float32, copy=False))
    if not frame_ids:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return np.concatenate(frame_ids), np.concatenate(boxes), np.concatenate(scores)


def letterbox_image(image: np.ndarray, image_size: int) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    ratio = min(image_size / height, image_size / width)
    resized_width, resized_height = round(width * ratio), round(height * ratio)
    pad_width, pad_height = image_size - resized_width, image_size - resized_height
    left, right = round(pad_width / 2 - 0.1), round(pad_width / 2 + 0.1)
    top, bottom = round(pad_height / 2 - 0.1), round(pad_height / 2 + 0.1)
    if (width, height) != (resized_width, resized_height):
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    if image.shape[:2] != (image_size, image_size):
        raise RuntimeError(f"letterbox shape mismatch: got {image.shape[:2]}, expected {(image_size, image_size)}")
    return image, ratio, left, top


def map_boxes_to_letterbox(boxes_tlbr: np.ndarray, ratio: float, left: int, top: int, image_size: int) -> np.ndarray:
    mapped = boxes_tlbr.astype(np.float32, copy=True)
    mapped[:, [0, 2]] = mapped[:, [0, 2]] * ratio + left
    mapped[:, [1, 3]] = mapped[:, [1, 3]] * ratio + top
    return np.clip(mapped, 0.0, float(image_size))


class FrozenYOLO11sROIExtractor:
    """Extract L2-normalized vectors from one frozen YOLO spatial layer."""

    def __init__(
        self,
        checkpoint: Path,
        ultralytics_root: Path,
        feature_layer: int,
        feature_stride: int,
        expected_channels: int,
        image_size: int,
        roi_output_size: int,
        device_name: str,
    ) -> None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
        if not ultralytics_root.is_dir():
            raise FileNotFoundError(f"missing Ultralytics source root: {ultralytics_root}")
        if device_name not in {"mps", "cpu"}:
            raise ValueError("device must be mps or cpu")
        sys.path.insert(0, str(ultralytics_root.resolve()))
        import torch
        import torch.nn.functional as functional
        from torchvision.ops import roi_align
        from ultralytics import YOLO

        if device_name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable; refusing an implicit CPU feature extraction")
        self.torch = torch
        self.functional = functional
        self.roi_align = roi_align
        self.device = torch.device(device_name)
        self.model = YOLO(str(checkpoint.resolve())).model.to(self.device).eval()
        self.feature_layer = int(feature_layer)
        self.feature_stride = int(feature_stride)
        self.expected_channels = int(expected_channels)
        self.image_size = int(image_size)
        self.roi_output_size = int(roi_output_size)
        self.captured: Any = None
        self.hook = self.model.model[self.feature_layer].register_forward_hook(self._capture)

    def _capture(self, _module: Any, _inputs: Any, output: Any) -> None:
        self.captured = output

    def extract(self, image_bgr: np.ndarray, boxes_tlbr: np.ndarray) -> np.ndarray:
        if boxes_tlbr.ndim != 2 or boxes_tlbr.shape[1] != 4 or len(boxes_tlbr) == 0:
            raise ValueError("extract expects a non-empty (N, 4) tlbr box matrix")
        image, ratio, left, top = letterbox_image(image_bgr, self.image_size)
        rgb = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))
        tensor = self.torch.from_numpy(rgb).to(self.device).float().div_(255.0).unsqueeze(0)
        mapped_boxes = map_boxes_to_letterbox(boxes_tlbr, ratio, left, top, self.image_size)
        rois = self.torch.cat(
            (
                self.torch.zeros((len(mapped_boxes), 1), dtype=self.torch.float32, device=self.device),
                self.torch.from_numpy(mapped_boxes).to(self.device),
            ),
            dim=1,
        )
        self.captured = None
        with self.torch.inference_mode():
            _ = self.model(tensor)
            if self.captured is None:
                raise RuntimeError(f"YOLO layer {self.feature_layer} was not captured")
            if self.captured.ndim != 4 or int(self.captured.shape[1]) != self.expected_channels:
                raise RuntimeError(
                    f"unexpected feature map shape {tuple(self.captured.shape)}; expected channel count {self.expected_channels}"
                )
            pooled = self.roi_align(
                self.captured,
                rois,
                output_size=(self.roi_output_size, self.roi_output_size),
                spatial_scale=1.0 / self.feature_stride,
                sampling_ratio=-1,
                aligned=True,
            )
            vector = pooled.mean(dim=(-1, -2))
            vector = self.functional.normalize(vector, p=2, dim=1)
        result = vector.detach().cpu().numpy().astype(np.float32, copy=False)
        if not np.isfinite(result).all():
            raise RuntimeError("feature extractor returned non-finite values")
        return result

    def close(self) -> None:
        self.hook.remove()


def extract_sequence_record(
    sequence_dir: Path,
    info: dict[str, Any],
    detections: dict[int, np.ndarray],
    extractor: FrozenYOLO11sROIExtractor,
) -> dict[str, np.ndarray]:
    frame_ids, boxes, scores, features = [], [], [], []
    image_dir = sequence_dir / str(info["image_dir"])
    for frame in range(1, int(info["frames"]) + 1):
        rows = detections.get(frame)
        if rows is None or len(rows) == 0:
            continue
        image_path = image_dir / f"{frame:06d}{info['image_extension']}"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"unable to read image: {image_path}")
        if image.shape[:2] != (int(info["height"]), int(info["width"])):
            raise RuntimeError(
                f"{image_path}: image shape {image.shape[:2]} does not match seqinfo "
                f"{(info['height'], info['width'])}"
            )
        vector = extractor.extract(image, rows[:, :4])
        if vector.shape != (len(rows), extractor.expected_channels):
            raise RuntimeError(f"{image_path}: unexpected ROI feature shape {vector.shape}")
        frame_ids.append(np.full(len(rows), frame, dtype=np.int32))
        boxes.append(rows[:, :4].astype(np.float32, copy=False))
        scores.append(rows[:, 4].astype(np.float32, copy=False))
        features.append(vector)
    if not frame_ids:
        return {
            "frame_ids": np.empty((0,), dtype=np.int32),
            "boxes_tlbr": np.empty((0, 4), dtype=np.float32),
            "scores": np.empty((0,), dtype=np.float32),
            "features": np.empty((0, extractor.expected_channels), dtype=np.float32),
        }
    return {
        "frame_ids": np.concatenate(frame_ids),
        "boxes_tlbr": np.concatenate(boxes),
        "scores": np.concatenate(scores),
        "features": np.concatenate(features),
    }


def save_sequence_record(path: Path, record: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite feature record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **record)


def load_sequence_record(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing feature record: {path}")
    with np.load(path, allow_pickle=False) as source:
        required = {"frame_ids", "boxes_tlbr", "scores", "features"}
        missing = sorted(required - set(source.files))
        if missing:
            raise ValueError(f"{path}: missing fields {', '.join(missing)}")
        record = {name: source[name] for name in required}
    if (
        record["frame_ids"].ndim != 1
        or record["boxes_tlbr"].ndim != 2
        or record["boxes_tlbr"].shape[1] != 4
        or record["scores"].ndim != 1
        or record["features"].ndim != 2
    ):
        raise ValueError(f"{path}: malformed feature-cache arrays")
    rows = len(record["frame_ids"])
    if len(record["boxes_tlbr"]) != rows or len(record["scores"]) != rows or len(record["features"]) != rows:
        raise ValueError(f"{path}: feature-cache row counts disagree")
    if record["features"].shape[1] == 0 or not np.isfinite(record["features"]).all():
        raise ValueError(f"{path}: invalid feature matrix")
    return {
        "frame_ids": record["frame_ids"].astype(np.int32, copy=False),
        "boxes_tlbr": record["boxes_tlbr"].astype(np.float32, copy=False),
        "scores": record["scores"].astype(np.float32, copy=False),
        "features": record["features"].astype(np.float32, copy=False),
    }


def assert_record_matches_detections(
    record: dict[str, np.ndarray], info: dict[str, Any], detections: dict[int, np.ndarray], path: Path
) -> dict[int, np.ndarray]:
    frame_ids, boxes, scores = flatten_detections(info, detections)
    if not np.array_equal(record["frame_ids"], frame_ids):
        raise ValueError(f"{path}: frame IDs do not match detector cache")
    if not np.array_equal(record["boxes_tlbr"], boxes):
        raise ValueError(f"{path}: boxes do not match detector cache")
    if not np.array_equal(record["scores"], scores):
        raise ValueError(f"{path}: scores do not match detector cache")
    norms = np.linalg.norm(record["features"], axis=1)
    if len(norms) and not np.allclose(norms, 1.0, rtol=2e-4, atol=2e-4):
        raise ValueError(f"{path}: ROI features are not L2-normalized")
    feature_by_frame: dict[int, np.ndarray] = {}
    for frame in range(1, int(info["frames"]) + 1):
        indices = np.flatnonzero(record["frame_ids"] == frame)
        if len(indices):
            feature_by_frame[frame] = record["features"][indices]
    return feature_by_frame
