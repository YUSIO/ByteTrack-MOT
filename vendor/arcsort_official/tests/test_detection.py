"""Tests for detector-wrapper class-id conventions (no detector models needed)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from arcsort.detection import RFDETRDetector, coco_category_to_index


def test_coco_category_to_index():
    # person, bicycle, the first gap (12), stop sign (13), toothbrush (90), out of range
    np.testing.assert_array_equal(
        coco_category_to_index(np.array([1, 2, 12, 13, 90, 91, -3])),
        [0, 1, -1, 11, 79, -1, -1],
    )


def test_matches_rfdetr_coco_table():
    coco = pytest.importorskip("rfdetr.assets.coco_classes")
    cats = sorted(coco.COCO_CLASSES)
    np.testing.assert_array_equal(coco_category_to_index(np.array(cats)), np.arange(80))


def test_rfdetr_wrapper_keeps_persons_and_feeds_rgb():
    seen = {}

    def predict(frame, threshold):
        seen["frame"] = frame
        return SimpleNamespace(
            xyxy=np.array([[0, 0, 10, 10], [5, 5, 9, 9]], dtype=float),
            confidence=np.array([0.9, 0.8]),
            class_id=np.array([1, 37]),  # person, sports ball (COCO category ids)
        )

    det = RFDETRDetector.__new__(RFDETRDetector)
    det.model = SimpleNamespace(predict=predict)
    det.conf_thresh, det.classes = 0.05, [0]
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr[..., 0] = 255  # blue in BGR
    out = det.detect(bgr)
    assert out.shape == (1, 6) and out[0, 5] == 0
    assert seen["frame"][0, 0].tolist() == [0, 0, 255]  # RGB: blue is the last channel
