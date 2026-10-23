"""Tests for arcsort/utils.py -- bounding box format conversions."""

from __future__ import annotations

import numpy as np
import pytest

from arcsort.utils import (
    cxcysr_to_xyxy,
    cxcywh_to_xyxy,
    xyxy_to_cxcysr,
    xyxy_to_cxcywh,
    xyxy_to_xywh,
    xywh_to_xyxy,
)


# ---------------------------------------------------------------------------
# xyxy <-> xywh
# ---------------------------------------------------------------------------

def test_xyxy_to_xywh_known_values():
    bbox = np.array([10.0, 20.0, 50.0, 80.0])
    result = xyxy_to_xywh(bbox)
    np.testing.assert_allclose(result, [10.0, 20.0, 40.0, 60.0])


def test_xywh_to_xyxy_known_values():
    bbox = np.array([10.0, 20.0, 40.0, 60.0])
    result = xywh_to_xyxy(bbox)
    np.testing.assert_allclose(result, [10.0, 20.0, 50.0, 80.0])


def test_xyxy_xywh_roundtrip():
    bbox = np.array([15.0, 25.0, 100.0, 200.0])
    result = xywh_to_xyxy(xyxy_to_xywh(bbox))
    np.testing.assert_allclose(result, bbox)


def test_xywh_xyxy_roundtrip():
    bbox = np.array([15.0, 25.0, 85.0, 175.0])
    result = xyxy_to_xywh(xywh_to_xyxy(bbox))
    np.testing.assert_allclose(result, bbox)


def test_xyxy_to_xywh_batch():
    bboxes = np.array([
        [0.0, 0.0, 10.0, 20.0],
        [5.0, 5.0, 15.0, 25.0],
    ])
    result = xyxy_to_xywh(bboxes)
    expected = np.array([
        [0.0, 0.0, 10.0, 20.0],
        [5.0, 5.0, 10.0, 20.0],
    ])
    np.testing.assert_allclose(result, expected)


def test_xywh_to_xyxy_batch():
    bboxes = np.array([
        [0.0, 0.0, 10.0, 20.0],
        [5.0, 5.0, 10.0, 20.0],
    ])
    result = xywh_to_xyxy(bboxes)
    expected = np.array([
        [0.0, 0.0, 10.0, 20.0],
        [5.0, 5.0, 15.0, 25.0],
    ])
    np.testing.assert_allclose(result, expected)


def test_xyxy_to_xywh_zero_size():
    bbox = np.array([50.0, 50.0, 50.0, 50.0])
    result = xyxy_to_xywh(bbox)
    np.testing.assert_allclose(result, [50.0, 50.0, 0.0, 0.0])


def test_xywh_to_xyxy_zero_size():
    bbox = np.array([50.0, 50.0, 0.0, 0.0])
    result = xywh_to_xyxy(bbox)
    np.testing.assert_allclose(result, [50.0, 50.0, 50.0, 50.0])


def test_xyxy_to_xywh_negative_coords():
    bbox = np.array([-10.0, -20.0, 10.0, 20.0])
    result = xyxy_to_xywh(bbox)
    np.testing.assert_allclose(result, [-10.0, -20.0, 20.0, 40.0])


def test_xyxy_to_xywh_large_values():
    bbox = np.array([0.0, 0.0, 1920.0, 1080.0])
    result = xyxy_to_xywh(bbox)
    np.testing.assert_allclose(result, [0.0, 0.0, 1920.0, 1080.0])


# ---------------------------------------------------------------------------
# xyxy <-> cxcywh
# ---------------------------------------------------------------------------

def test_xyxy_to_cxcywh_known_values():
    bbox = np.array([10.0, 20.0, 50.0, 80.0])
    result = xyxy_to_cxcywh(bbox)
    np.testing.assert_allclose(result, [30.0, 50.0, 40.0, 60.0])


def test_cxcywh_to_xyxy_known_values():
    bbox = np.array([30.0, 50.0, 40.0, 60.0])
    result = cxcywh_to_xyxy(bbox)
    np.testing.assert_allclose(result, [10.0, 20.0, 50.0, 80.0])


def test_xyxy_cxcywh_roundtrip():
    bbox = np.array([15.0, 25.0, 100.0, 200.0])
    result = cxcywh_to_xyxy(xyxy_to_cxcywh(bbox))
    np.testing.assert_allclose(result, bbox)


def test_xyxy_to_cxcywh_batch():
    bboxes = np.array([
        [0.0, 0.0, 10.0, 20.0],
        [100.0, 200.0, 150.0, 250.0],
    ])
    result = xyxy_to_cxcywh(bboxes)
    expected = np.array([
        [5.0, 10.0, 10.0, 20.0],
        [125.0, 225.0, 50.0, 50.0],
    ])
    np.testing.assert_allclose(result, expected)


def test_cxcywh_to_xyxy_zero_size():
    bbox = np.array([50.0, 50.0, 0.0, 0.0])
    result = cxcywh_to_xyxy(bbox)
    np.testing.assert_allclose(result, [50.0, 50.0, 50.0, 50.0])


# ---------------------------------------------------------------------------
# xyxy <-> cxcysr
# ---------------------------------------------------------------------------

def test_xyxy_to_cxcysr_known_values():
    bbox = np.array([10.0, 20.0, 50.0, 80.0])
    result = xyxy_to_cxcysr(bbox)
    # cx=30, cy=50, s=40*60=2400, r=40/60=0.6667
    assert result[0] == pytest.approx(30.0)
    assert result[1] == pytest.approx(50.0)
    assert result[2] == pytest.approx(2400.0)
    assert result[3] == pytest.approx(40.0 / 60.0, abs=1e-4)


def test_cxcysr_to_xyxy_known_values():
    # cx=30, cy=50, s=2400, r=40/60
    bbox = np.array([30.0, 50.0, 2400.0, 40.0 / 60.0])
    result = cxcysr_to_xyxy(bbox)
    np.testing.assert_allclose(result, [10.0, 20.0, 50.0, 80.0], atol=0.1)


def test_xyxy_cxcysr_roundtrip():
    bbox = np.array([10.0, 20.0, 110.0, 220.0])
    result = cxcysr_to_xyxy(xyxy_to_cxcysr(bbox))
    np.testing.assert_allclose(result, bbox, atol=0.1)


def test_xyxy_to_cxcysr_batch():
    bboxes = np.array([
        [0.0, 0.0, 10.0, 20.0],
        [100.0, 200.0, 150.0, 250.0],
    ])
    result = xyxy_to_cxcysr(bboxes)
    assert result.shape == (2, 4)
    # First box: cx=5, cy=10, s=200, r=10/20=0.5
    assert result[0, 0] == pytest.approx(5.0)
    assert result[0, 1] == pytest.approx(10.0)
    assert result[0, 2] == pytest.approx(200.0)
    assert result[0, 3] == pytest.approx(0.5, abs=1e-4)


def test_xyxy_to_cxcysr_zero_height():
    """Zero-height box uses eps to avoid division by zero."""
    bbox = np.array([10.0, 20.0, 50.0, 20.0])
    result = xyxy_to_cxcysr(bbox)
    # h=0, so s=0, r=w/(0+eps) which is very large
    assert result[2] == pytest.approx(0.0)
    assert np.isfinite(result[3])


def test_cxcysr_to_xyxy_zero_area():
    """Zero-area box should produce zero-size bbox without NaN."""
    bbox = np.array([50.0, 50.0, 0.0, 1.0])
    result = cxcysr_to_xyxy(bbox)
    assert np.all(np.isfinite(result))


def test_xyxy_to_cxcysr_large_box():
    bbox = np.array([0.0, 0.0, 1920.0, 1080.0])
    result = xyxy_to_cxcysr(bbox)
    assert result[0] == pytest.approx(960.0)
    assert result[1] == pytest.approx(540.0)
    assert result[2] == pytest.approx(1920.0 * 1080.0)


def test_xyxy_to_xywh_does_not_mutate_input():
    bbox = np.array([10.0, 20.0, 50.0, 80.0])
    original = bbox.copy()
    xyxy_to_xywh(bbox)
    np.testing.assert_array_equal(bbox, original)


def test_xyxy_to_cxcywh_does_not_mutate_input():
    bbox = np.array([10.0, 20.0, 50.0, 80.0])
    original = bbox.copy()
    xyxy_to_cxcywh(bbox)
    np.testing.assert_array_equal(bbox, original)


def test_xyxy_to_cxcysr_does_not_mutate_input():
    bbox = np.array([10.0, 20.0, 50.0, 80.0])
    original = bbox.copy()
    xyxy_to_cxcysr(bbox)
    np.testing.assert_array_equal(bbox, original)
