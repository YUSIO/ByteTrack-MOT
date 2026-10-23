import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from fctrack.spatial_intervention import (
    FIXED_PROTOCOL, QUERY_COLUMNS, _load_scorer, apply, crop_views,
    extract, intervention_masks, scores_from_logits,
)


class SpatialInterventionTests(unittest.TestCase):
    def make_fixture(self, temporary, files):
        root = Path(temporary).resolve()
        (root / "reference").mkdir()
        for name, content in files.items():
            (root / "reference" / f"{name}.txt").write_bytes(content)
            seq = root / "data/train" / name
            (seq / "img1").mkdir(parents=True)
            (seq / "gt").mkdir()
            (seq / "gt/gt.txt").write_text("GT MUST NOT BE READ")
            (seq / "seqinfo.ini").write_text("[Sequence]\nseqLength=3\nimDir=img1\nimExt=.jpg\n")
            for frame in (1, 2, 3):
                (seq / "img1" / f"{frame:06d}.jpg").write_bytes(bytes([frame]))
        (root / "seed.pt").write_bytes(b"fixture weights")
        config = root / "config.json"
        config.write_text(json.dumps(dict(
            spatial_intervention=FIXED_PROTOCOL,
            cells=dict(dev=dict(dataset_root="data", split="train", reference_tracks="reference",
                                verifiers=dict(seed=dict(kind="seed", weights="seed.pt")))))))
        return root, config

    def run_extract(self, root, config, logits, name="extract"):
        calls, reads = [], []
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            if "gt" in path.parts:
                raise AssertionError(f"GT read: {path}")
            reads.append(str(path))
            return original_open(path, *args, **kwargs)

        def loader(weights, device):
            self.assertEqual(weights, root / "seed.pt")
            self.assertEqual(device, "cpu")

            def score(image, boxes):
                frame = int(image[0, 0, 0])
                calls.append((frame, boxes.copy()))
                # Scorer never receives the immutable source-row objects.
                boxes[:] = 99999
                return np.asarray(logits[frame], dtype=float)
            return score

        with patch.object(Path, "open", guarded_open):
            result = extract(root, config, "dev", root / name, "cpu", scorer_loader=loader,
                             image_decoder=lambda b, _: np.full((2, 2, 3), b[0], dtype=np.uint8))
        return result, calls, reads

    def test_mask_area_positions_shared_original_median_and_input_immutable(self):
        x = np.linspace(0, 1, 2 * 3 * 64 * 64, dtype=np.float32).reshape(2, 3, 64, 64)
        before = x.copy()
        masks = intervention_masks()
        self.assertEqual(list(masks), ["center", "tl", "tr", "bl", "br"])
        for mask in masks.values():
            self.assertEqual(int(mask.sum()), 441)
        self.assertEqual(int(np.stack(list(masks.values())).sum(axis=0).max()), 1)
        expected_center = np.zeros((64, 64), bool)
        expected_center[22:43, 22:43] = True
        np.testing.assert_array_equal(masks["center"], expected_center)
        medians = np.median(before, axis=(2, 3))
        views = list(crop_views(x))
        self.assertEqual(len(views), 6)
        np.testing.assert_array_equal(views[0][1], before)
        for name, value in views[1:]:
            mask = masks[name]
            np.testing.assert_array_equal(value[:, :, ~mask], before[:, :, ~mask])
            np.testing.assert_array_equal(value[:, :, mask], np.broadcast_to(medians[:, :, None], (2, 3, 441)))
        np.testing.assert_array_equal(x, before)

    def test_original_logit_and_six_real_model_batches_ignore_regression(self):
        import torch
        from fctrack.verifier import scalars, seed_crop

        inputs, scales = [], []

        class Model:
            def __call__(self, x, scale):
                inputs.append(x.cpu().numpy().copy())
                scales.append(scale.cpu().numpy().copy())
                value = 1000 + x.mean((1, 2, 3))
                # Irrelevant regression outputs are deliberately invalid.
                return torch.column_stack((value, *[torch.full_like(value, float("nan")) for _ in range(4)]))

        image = np.arange(96 * 96 * 3, dtype=np.uint8).reshape(96, 96, 3)
        boxes = np.array([[20., 30., 12., 18.], [0., 0., 8., 8.]])
        with patch("fctrack.verifier.load", return_value=Model()):
            values = _load_scorer(Path("unused"), "cpu")(image, boxes)
        self.assertEqual(values.shape, (2, 6))
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values > 999).all())
        self.assertEqual(len(inputs), 6)
        side = np.sqrt(boxes[:, 2] * boxes[:, 3])
        expected = np.stack([seed_crop(image, *(b[:2] + b[2:] / 2), s) for b, s in zip(boxes, side)])
        np.testing.assert_array_equal(inputs[0], expected)
        for (_, expected_view), actual in zip(crop_views(expected), inputs):
            np.testing.assert_array_equal(actual, expected_view)
        for value in scales:
            np.testing.assert_array_equal(value, scalars(side, np.full(2, 96.)))

    def test_extract_and_apply_strict_ties_exact_subsequence_empty_and_no_gt(self):
        with TemporaryDirectory() as temporary:
            rows = [b"1,7,1.20,2.30,3.4,4.5,0.9100,-1,-1,-1\r\n",
                    b"1,3,11,12,13,14,0.900,-1,-1,-1\n",
                    b"3,7,5.00,6,7,8,0.7,-1,-1,-1"]
            root, config = self.make_fixture(temporary, {"UAVSwarm-01": b"".join(rows), "UAVSwarm-03": b""})
            logits = {1: [[0, 0, 1, 1, -1, -1], [-1, 1, 1, 1, 1, 1]], 3: [[1, 2, 0, 0, 0, 0]]}
            summary, calls, reads = self.run_extract(root, config, logits)
            self.assertEqual([f for f, _ in calls], [1, 3])
            self.assertEqual(summary["frame_count"], 6)
            self.assertEqual(summary["scored_frame_count"], 2)
            self.assertEqual(summary["model_forward_calls"], 12)
            self.assertEqual(summary["row_views"], 18)
            self.assertFalse(any(p.endswith("000002.jpg") for p in reads))
            self.assertFalse(summary["gt_used"])
            with (root / "extract/queries/UAVSwarm-01.csv").open() as f:
                q = list(csv.DictReader(f))
            self.assertEqual([int(r["row_index"]) for r in q], [0, 1, 2])
            self.assertEqual(float(q[0]["x"]), 1.2)
            np.testing.assert_array_equal([float(q[0][k]) for k in ("raw", "paired", "sham")], [0, 0, 2])
            threshold = root / "thresholds.json"
            threshold.write_text(json.dumps(dict(schema_version=1, thresholds=dict(raw=0, paired=0, sham=0))))
            original_open = Path.open

            def apply_guard(path, *args, **kwargs):
                if "gt" in path.parts or path.suffix in (".jpg", ".pt"):
                    raise AssertionError(f"forbidden apply read: {path}")
                return original_open(path, *args, **kwargs)

            with patch.object(Path, "open", apply_guard):
                result = apply(root / "extract", threshold, root / "apply")
            for arm, retained in (("A", rows), ("A.raw", [rows[0], rows[2]]),
                                  ("A.paired", rows[:2]), ("A.sham", rows)):
                self.assertEqual((root / "apply/tracks" / arm / "UAVSwarm-01.txt").read_bytes(), b"".join(retained))
                self.assertEqual((root / "apply/tracks" / arm / "UAVSwarm-03.txt").read_bytes(), b"")
            self.assertEqual(result["rejected_rows"], dict(raw=1, paired=1, sham=0))
            self.assertEqual(result["input_rows"], 3)
            self.assertFalse(result["model_called"])
            self.assertFalse(result["threshold_fitted"])

    def test_empty_all_sequences_and_all_rejected(self):
        for empty in (True, False):
            with self.subTest(empty=empty), TemporaryDirectory() as temporary:
                content = b"" if empty else b"1,1,1,2,3,4,.8,-1,-1,-1\n"
                root, config = self.make_fixture(temporary, {"UAVSwarm-01": content})
                summary, calls, _ = self.run_extract(root, config, {1: [[0, 0, 0, 0, 0, 0]]})
                thresholds = root / "thresholds.json"
                thresholds.write_text(json.dumps(dict(schema_version=1, thresholds=dict(raw=1, paired=1, sham=1))))
                result = apply(root / "extract", thresholds, root / "applied")
                self.assertEqual(len(calls), 0 if empty else 1)
                self.assertEqual(result["input_rows"], 0 if empty else 1)
                for arm in ("raw", "paired", "sham"):
                    self.assertEqual((root / f"applied/tracks/A.{arm}/UAVSwarm-01.txt").read_bytes(), b"")

    def test_no_overwrite_before_model_or_threshold_read(self):
        with TemporaryDirectory() as temporary:
            root, config = self.make_fixture(temporary, {"UAVSwarm-01": b""})
            output = root / "exists"
            output.mkdir()
            with self.assertRaises(FileExistsError), patch("fctrack.spatial_intervention._load_scorer") as loader:
                extract(root, config, "dev", output, "cpu")
            loader.assert_not_called()
            with self.assertRaises(FileExistsError):
                apply(root / "missing-extraction", root / "missing-thresholds", output)

    def test_future_rows_do_not_change_prefix_query(self):
        queries = []
        for suffix_id, suffix_logit in ((9, 5), (99, -5)):
            with TemporaryDirectory() as temporary:
                prefix = b"1,9,1,2,3,4,.8,-1,-1,-1\n"
                root, config = self.make_fixture(temporary, {"UAVSwarm-01": prefix +
                    f"3,{suffix_id},5,6,7,8,.7,-1,-1,-1\n".encode()})
                self.run_extract(root, config, {1: [[1, 2, 3, 4, 5, 6]], 3: [[suffix_logit] * 6]})
                with (root / "extract/queries/UAVSwarm-01.csv").open() as f:
                    queries.append(next(csv.DictReader(f)))
        self.assertEqual(queries[0], queries[1])

    def test_invalid_frame_order_or_nonfinite_logits_abort_without_summary(self):
        for kind in ("order", "nan", "shape"):
            with self.subTest(kind=kind), TemporaryDirectory() as temporary:
                content = (b"3,1,1,2,3,4,.8,-1,-1,-1\n1,1,1,2,3,4,.8,-1,-1,-1\n" if kind == "order"
                           else b"1,1,1,2,3,4,.8,-1,-1,-1\n")
                root, config = self.make_fixture(temporary, {"UAVSwarm-01": content})
                logits = {1: [[float("nan")] * 6] if kind == "nan" else [[1] * 5], 3: [[1] * 6]}
                with self.assertRaises(ValueError):
                    self.run_extract(root, config, logits)
                self.assertFalse((root / "extract/summary.json").exists())

    def test_tampered_extraction_rejected_before_output_creation(self):
        with TemporaryDirectory() as temporary:
            root, config = self.make_fixture(temporary, {"UAVSwarm-01": b"1,1,1,2,3,4,.8,-1,-1,-1\n"})
            self.run_extract(root, config, {1: [[1, 2, 3, 4, 5, 6]]})
            thresholds = root / "thresholds.json"
            thresholds.write_text(json.dumps(dict(schema_version=1, thresholds=dict(raw=0, paired=0, sham=0))))
            with (root / "extract/queries/UAVSwarm-01.csv").open("a") as f:
                f.write("tampered\n")
            with self.assertRaisesRegex(RuntimeError, "extraction input changed"):
                apply(root / "extract", thresholds, root / "applied")
            self.assertFalse((root / "applied").exists())


if __name__ == "__main__":
    unittest.main()
