import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from fctrack.output_verification import FIXED_PROTOCOL, verify_output_tracks


class OutputVerificationTests(unittest.TestCase):
    def make_fixture(self, root, files):
        root = Path(root)
        (root / "references").mkdir()
        for name, content in files.items():
            (root / "references" / f"{name}.txt").write_bytes(content)
            seq = root / "dataset" / "train" / name
            (seq / "img1").mkdir(parents=True)
            (seq / "gt").mkdir()
            (seq / "gt" / "gt.txt").write_text("MUST NEVER BE READ")
            (seq / "seqinfo.ini").write_text(
                "[Sequence]\nseqLength=3\nimDir=img1\nimExt=.jpg\n")
            # Stub decoder below makes the current-frame image identity explicit.
            for frame in (1, 2, 3):
                (seq / "img1" / f"{frame:06d}.jpg").write_bytes(bytes([frame]))
        (root / "weights.pt").write_bytes(b"stub original seed weights")
        config = dict(cells=dict(dev=dict(dataset_root="dataset", split="train",
                      reference_tracks="references", verifiers=dict(seed=dict(
                          weights="weights.pt", kind="seed")))),
                      output_verification=dict(FIXED_PROTOCOL))
        path = root / "config.json"
        path.write_text(json.dumps(config))
        return root, path

    def run_fixture(self, root, config, probabilities, output="output"):
        calls, reads = [], []
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            if "gt" in path.parts:
                raise AssertionError(f"GT access: {path}")
            reads.append(str(path))
            return original_open(path, *args, **kwargs)

        def loader(kind, weights, device):
            self.assertEqual((kind, device), ("seed", "cpu"))

            def verify(image, boxes):
                frame = int(image[0, 0, 0])
                calls.append((frame, boxes.copy()))
                # Prove the saved geometry cannot be rewritten by the model.
                boxes[:] = 99999
                return np.asarray(probabilities[frame], float), object()
            return verify

        with patch.object(Path, "open", guarded_open):
            summary = verify_output_tracks(
                root, config, "dev", root / output, "cpu", verifier_loader=loader,
                image_decoder=lambda raw, _: np.full((2, 2, 3), raw[0], dtype=np.uint8))
        return summary, calls, reads

    def test_exact_source_subset_all_rows_threshold_equality_and_no_gt(self):
        with TemporaryDirectory() as tmp:
            rows = [b"1,7,1.20,2.30,3.4,4.5,0.9100,-1,-1,-1\r\n",
                    b"1,3,11,12,13,14,0.900,-1,-1,-1\n",
                    b"3,7,5.00,6,7,8,0.7,-1,-1,-1"]
            root, config = self.make_fixture(tmp, {"UAVSwarm-01": b"".join(rows)})
            summary, calls, reads = self.run_fixture(root, config, {1: [.97, .969999], 3: [1.]})
            self.assertEqual((root / "output/tracks/A/UAVSwarm-01.txt").read_bytes(), b"".join(rows))
            self.assertEqual((root / "output/tracks/A.verify/UAVSwarm-01.txt").read_bytes(), rows[0] + rows[2])
            self.assertEqual([frame for frame, _ in calls], [1, 3])
            np.testing.assert_array_equal(calls[0][1], [[1.2, 2.3, 3.4, 4.5], [11, 12, 13, 14]])
            np.testing.assert_array_equal(calls[1][1], [[5, 6, 7, 8]])
            with (root / "output/queries/UAVSwarm-01.csv").open() as stream:
                queries = list(csv.DictReader(stream))
            self.assertEqual([q["row_index"] for q in queries], ["0", "1", "2"])
            self.assertEqual([q["id"] for q in queries], ["7", "3", "7"])
            self.assertEqual([q["keep"] for q in queries], ["1", "0", "1"])
            self.assertEqual(float(queries[0]["x"]), 1.2)
            self.assertEqual((summary["input_rows"], summary["kept_rows"], summary["rejected_rows"]), (3, 2, 1))
            self.assertFalse(any(path.endswith("000002.jpg") for path in reads))
            self.assertFalse(summary["gt_used"])
            for path, digest in summary["input_sha256"].items():
                self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)
            for relative, digest in summary["output_sha256"].items():
                self.assertEqual(hashlib.sha256((root / "output" / relative).read_bytes()).hexdigest(), digest)

    def test_all_rejected_and_empty_sequence_create_empty_track_files(self):
        with TemporaryDirectory() as tmp:
            root, config = self.make_fixture(tmp, {
                "UAVSwarm-01": b"1,1,1,2,3,4,0.99,-1,-1,-1\n", "UAVSwarm-03": b""})
            summary, calls, _ = self.run_fixture(root, config, {1: [0.]})
            for name in ("UAVSwarm-01", "UAVSwarm-03"):
                self.assertEqual((root / "output/tracks/A.verify" / f"{name}.txt").read_bytes(), b"")
            self.assertEqual((root / "output/tracks/A/UAVSwarm-03.txt").read_bytes(), b"")
            self.assertEqual(len(calls), 1)
            self.assertEqual(summary["per_sequence"]["UAVSwarm-03"]["input_rows"], 0)

    def test_prefix_decisions_do_not_depend_on_future_rows_or_images(self):
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            prefix = b"1,9,1,2,3,4,.83,-1,-1,-1\n"
            summaries = []
            for temporary, suffix, probability in (
                    (first, b"3,9,9,9,9,9,.99,-1,-1,-1\n", 1.),
                    (second, b"3,123,2,2,2,2,.71,-1,-1,-1\n", 0.)):
                root, config = self.make_fixture(temporary, {"UAVSwarm-01": prefix + suffix})
                self.run_fixture(root, config, {1: [.97], 3: [probability]})
                with (root / "output/queries/UAVSwarm-01.csv").open() as stream:
                    summaries.append(next(csv.DictReader(stream)))
            self.assertEqual(summaries[0], summaries[1])

    def test_invalid_probabilities_abort_without_completed_summary(self):
        with TemporaryDirectory() as tmp:
            root, config = self.make_fixture(tmp, {
                "UAVSwarm-01": b"1,1,1,2,3,4,.9,-1,-1,-1\n"})
            with self.assertRaisesRegex(ValueError, "invalid verifier probabilities"):
                self.run_fixture(root, config, {1: [float("nan")]})
            self.assertFalse((root / "output/summary.json").exists())


if __name__ == "__main__":
    unittest.main()
