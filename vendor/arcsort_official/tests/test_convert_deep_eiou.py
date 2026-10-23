"""Tests for tools/convert_deep_eiou.py -- Deep-EIoU asset conversion.

The critical property: after converting detections to MOT txt and
embeddings to npz, the boxes the tracker loads from the txt must hit the
embedding-cache keys exactly (otherwise every lookup silently misses and
ReID runs on zero embeddings).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from arcsort.reid import EmbeddingComputer
from tools.convert_deep_eiou import convert_sequence, quantize_box
from tools.run_cached import load_detections


def _fake_assets(tmp_path, num_frames=3, dets_per_frame=4, emb_dim=512):
    rng = np.random.RandomState(0)
    det_frames = []
    emb_frames = []
    for _ in range(num_frames):
        x1 = rng.uniform(0, 1200, dets_per_frame)
        y1 = rng.uniform(0, 600, dets_per_frame)
        w = rng.uniform(30, 120, dets_per_frame)
        h = rng.uniform(60, 220, dets_per_frame)
        obj_conf = rng.uniform(0.5, 1.0, dets_per_frame)
        cls_conf = rng.uniform(0.5, 1.0, dets_per_frame)
        dets = np.column_stack([
            x1, y1, x1 + w, y1 + h, obj_conf, cls_conf,
            np.zeros(dets_per_frame),
        ]).astype(np.float32)
        det_frames.append(dets)
        embs = rng.randn(dets_per_frame, 1, emb_dim).astype(np.float32)
        emb_frames.append(embs)

    det_file = tmp_path / "detection" / "seq001.npy"
    emb_file = tmp_path / "embedding" / "seq001.npy"
    det_file.parent.mkdir(parents=True)
    emb_file.parent.mkdir(parents=True)
    np.save(det_file, np.array(det_frames, dtype=object), allow_pickle=True)
    np.save(emb_file, np.array(emb_frames, dtype=object), allow_pickle=True)
    return det_file, emb_file


def test_quantize_box_round_trip():
    x1, y1, x2, y2 = 489.7512, 380.2549, 594.0031, 543.0017
    x1q, y1q, x2q, y2q = quantize_box(x1, y1, x2, y2)
    # Matches txt semantics: x1, w formatted to 2 decimals, x2 = x1 + w
    assert x1q == pytest.approx(round(x1, 2))
    assert x2q == pytest.approx(round(x1, 2) + round(x2 - x1, 2))


def test_convert_and_load_detections(tmp_path):
    det_file, emb_file = _fake_assets(tmp_path)
    out_det = tmp_path / "out" / "seq001.txt"
    out_emb = tmp_path / "out_emb" / "seq001.npz"
    n_dets, n_embs = convert_sequence(det_file, emb_file, out_det, out_emb)
    assert n_dets == 12 and n_embs == 12

    dets_by_frame = load_detections(out_det)
    assert set(dets_by_frame) == {1, 2, 3}
    # Scores are obj_conf * cls_conf
    raw = np.load(det_file, allow_pickle=True)
    frame0 = np.asarray(raw[0], dtype=np.float64)
    expected = np.sort(frame0[:, 4] * frame0[:, 5])
    got = np.sort(dets_by_frame[1][:, 4].astype(np.float64))
    np.testing.assert_allclose(got, expected, atol=1e-4)


def test_converted_boxes_hit_embedding_cache(tmp_path):
    """Every box loaded from the txt must find its embedding in the npz."""
    det_file, emb_file = _fake_assets(tmp_path)
    out_det = tmp_path / "out" / "seq001.txt"
    out_emb = tmp_path / "out_emb" / "seq001.npz"
    convert_sequence(det_file, emb_file, out_det, out_emb)

    computer = EmbeddingComputer()
    computer.load_cache(out_emb)
    dets_by_frame = load_detections(out_det)

    for frame_id, dets in dets_by_frame.items():
        computer.set_frame(frame_id)
        embs = computer.compute_embeddings(None, dets[:, :4])
        # A cache miss returns a zero row; all rows must be unit-norm hits
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-3,
                                   err_msg=f"cache miss at frame {frame_id}")


def test_convert_missing_embeddings_ok(tmp_path):
    det_file, _ = _fake_assets(tmp_path)
    out_det = tmp_path / "out" / "seq001.txt"
    n_dets, n_embs = convert_sequence(det_file, None, out_det, None)
    assert n_dets == 12 and n_embs == 0
    assert out_det.exists()


def test_convert_rescales_to_image_coordinates(tmp_path):
    """Deep-EIoU boxes (network-input space) must map back to image pixels
    when a rescale factor is given: 1280x720 image -> scale 1.111..."""
    det_file, _ = _fake_assets(tmp_path)
    out_scaled = tmp_path / "out_scaled" / "seq001.txt"
    out_raw = tmp_path / "out_raw" / "seq001.txt"
    scale = min(1440 / 1280, 800 / 720)
    convert_sequence(det_file, None, out_raw, None, rescale=1.0)
    convert_sequence(det_file, None, out_scaled, None, rescale=scale)

    raw = load_detections(out_raw)[1]
    scaled = load_detections(out_scaled)[1]
    # Coordinates shrink by the scale factor; scores unchanged
    np.testing.assert_allclose(scaled[:, :4] * scale, raw[:, :4], atol=0.05)
    np.testing.assert_allclose(scaled[:, 4], raw[:, 4], atol=1e-6)
