"""Boundary, causal-history and immutable-cache checks for dense proposals."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from prebirth.infer import decode_batch, export_sequence, frame_bounds, nms, tile_coordinates
from prebirth.merge import merge_sequence, parse_rows, select_additions


class PrebirthInferenceTests(unittest.TestCase):
    def test_tiles_cover_every_original_pixel_and_do_not_resize(self):
        for width, height in ((31, 17), (512, 512), (1088, 1920), (513, 641)):
            coverage = np.zeros((height, width), dtype=np.uint8)
            coords = tile_coordinates(width, height, 512, 64)
            for x, y, w, h in coords:
                coverage[y:y+h, x:x+w] += 1
                self.assertLessEqual(w, 512)
                self.assertLessEqual(h, 512)
            self.assertTrue((coverage > 0).all())
        self.assertEqual(frame_bounds(7, "train"), (1, 4))
        self.assertEqual(frame_bounds(7, "dev"), (5, 7))
        self.assertEqual(frame_bounds(164, "train"), (1, 83))
        self.assertEqual(frame_bounds(164, "dev"), (84, 164))

    def test_decode_uses_raw_offset_and_removes_padding_before_budget(self):
        heat = torch.full((1, 1, 4, 4), -20.0)
        heat[0, 0, 1, 1] = 2.0
        heat[0, 0, 3, 3] = 8.0  # Higher-scoring peak outside valid tile support.
        offset = torch.zeros((1, 2, 4, 4))
        offset[0, :, 1, 1] = torch.tensor((0.25, 0.5))
        predictions = {"heatmap": heat, "offset": offset,
                       "log_wh": torch.full_like(offset, math.log(4.0))}
        rows, counts = decode_batch(predictions, [(0, 0, 9, 9)], (9, 9), 0.1, 1, 16, 2, 0)
        self.assertEqual(len(rows), 1)
        np.testing.assert_allclose(rows[0, :4], (3, 4, 7, 8))
        self.assertEqual(counts["invalid_or_padding"], 1)
        self.assertEqual(counts["tile_budget_candidates"], 1)
        self.assertAlmostEqual(rows[0, 4], torch.sigmoid(torch.tensor(2.)).item())

    def test_adjacent_padding_peak_cannot_suppress_valid_peak(self):
        heat = torch.full((1, 1, 4, 4), -20.0)
        heat[0, 0, 1, 1] = 2.0  # Valid decoded center (4, 4).
        heat[0, 0, 2, 1] = 8.0  # Adjacent center (4, 8), outside height 8.
        offset = torch.zeros((1, 2, 4, 4))
        predictions = {"heatmap": heat, "offset": offset,
                       "log_wh": torch.full_like(offset, math.log(4.0))}
        rows, counts = decode_batch(predictions, [(0, 0, 16, 8)], (8, 16), .1, 1, 16, 2, 0)
        np.testing.assert_array_equal(rows[0, :4], (2, 2, 6, 6))
        self.assertEqual(len(rows), 1)
        self.assertEqual(counts["threshold_localmax"], 1)
        self.assertEqual(counts["invalid_or_padding"], 1)
        self.assertEqual(counts["rescued_localmax"], 1)

    def test_peak_validity_uses_decoded_center_instead_of_grid_index(self):
        for high_grid_y, high_offset_y, valid in ((2, -.5, True), (1, 1., False)):
            with self.subTest(high_grid_y=high_grid_y):
                heat = torch.full((1, 1, 4, 4), -20.0)
                low_grid_y = high_grid_y - 1
                heat[0, 0, low_grid_y, 1] = 2.
                heat[0, 0, high_grid_y, 1] = 8.
                offset = torch.zeros((1, 2, 4, 4))
                offset[0, 1, low_grid_y, 1] = .5
                offset[0, 1, high_grid_y, 1] = high_offset_y
                predictions = {"heatmap": heat, "offset": offset,
                               "log_wh": torch.full_like(offset, math.log(4.))}
                rows, counts = decode_batch(predictions, [(0, 0, 16, 8)], (8, 16), .1, 100, 16, 2, 0)
                self.assertEqual(len(rows), 1)
                expected_score = torch.sigmoid(torch.tensor(8. if valid else 2.)).item()
                self.assertEqual(rows[0, 4], expected_score)
                self.assertEqual(counts["rescued_localmax"], int(not valid))

    def test_invalid_geometry_cannot_suppress_valid_neighbor(self):
        for failure in ("clipped_too_small", "nonfinite_center", "nonfinite_size"):
            with self.subTest(failure=failure):
                heat = torch.full((1, 1, 4, 4), -20.)
                heat[0, 0, 1, 1] = 2.
                heat[0, 0, 1, 0] = 8.
                offset = torch.zeros((1, 2, 4, 4))
                log_wh = torch.full_like(offset, math.log(4.))
                if failure == "clipped_too_small":
                    log_wh[0, :, 1, 0] = math.log(2.)  # Center x=0; clipped width=1.
                elif failure == "nonfinite_center":
                    offset[0, 0, 1, 0] = float("nan")
                else:
                    log_wh[0, 0, 1, 0] = float("inf")
                rows, counts = decode_batch({"heatmap": heat, "offset": offset, "log_wh": log_wh},
                                            [(0, 0, 16, 16)], (16, 16), .1, 100, 16, 2, 0)
                self.assertEqual(len(rows), 1)
                np.testing.assert_array_equal(rows[0, :4], (2, 2, 6, 6))
                self.assertEqual(counts["invalid_or_padding"], 1)
                self.assertEqual(counts["rescued_localmax"], 1)

    def test_all_valid_decode_matches_original_sparse_reference_exactly(self):
        generator = torch.Generator().manual_seed(138)
        outputs = {"heatmap": torch.randn((2, 1, 8, 8), generator=generator),
                   "offset": torch.full((2, 2, 8, 8), .5),
                   "log_wh": (torch.rand((2, 2, 8, 8), generator=generator) * 8 + 4).log()}
        coordinates = [(0, 0, 32, 32), (11, 7, 32, 32)]
        heat = outputs["heatmap"].sigmoid()
        original_peaks = heat.eq(F.max_pool2d(heat, 3, 1, 1)) & (heat >= .2)
        for budget in (1, 7, 128):
            with self.subTest(budget=budget):
                # Legacy sparse arithmetic for this all-valid fixture. This checks
                # bitwise coordinates/scores/order independently of dense decoding.
                expected = []
                for b, (left, top, _, _) in enumerate(coordinates):
                    ys, xs = torch.where(original_peaks[b, 0])
                    center = (torch.stack((xs, ys), 1).float() + outputs["offset"][b, :, ys, xs].T) * 4.
                    wh = outputs["log_wh"][b, :, ys, xs].T.clamp(math.log(2.), math.log(32.)).exp()
                    center = center + center.new_tensor((left, top))
                    boxes = torch.cat((center - wh / 2, center + wh / 2), 1)
                    boxes[:, 0::2].clamp_(0, 43)
                    boxes[:, 1::2].clamp_(0, 39)
                    scores = heat[b, 0, ys, xs]
                    if len(scores) > budget:
                        selected = torch.argsort(scores, descending=True, stable=True)[:budget]
                        boxes, scores = boxes[selected], scores[selected]
                    expected.extend(torch.cat((boxes, scores[:, None]), 1).numpy())
                rows, counts = decode_batch(outputs, coordinates, (39, 43), .2, budget, 32, 2, 0)
                np.testing.assert_array_equal(rows, np.asarray(expected))
                self.assertEqual(counts["invalid_or_padding"], 0)
                self.assertEqual(counts["rescued_localmax"], 0)

    def test_tile_duplicates_suppressed_without_losing_other_objects(self):
        proposals = [[1, 1, 9, 9, .8], [1.1, 1.1, 9.1, 9.1, .9], [20, 20, 28, 28, .7]]
        kept = nms(proposals, .5)
        self.assertEqual(len(kept), 2)
        np.testing.assert_allclose(kept[:, 4], (.9, .7))

    def test_merge_preserves_original_bytes_and_raw_proposal_score(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "base.txt"
            proposal = root / "proposal.txt"
            # Final newline omitted intentionally; precision and spaces must survive.
            original = b"1,-1,1.000000, 1,4,4,0.923456,-1,-1,-1"
            base.write_bytes(original)
            proposal.write_text("1,-1,1,1,4,4,0.99,-1,-1,-1\n"
                                "1,-1,12,12,4,4,0.87654321,-1,-1,-1\n"
                                "1,-1,12.1,12.1,4,4,0.85,-1,-1,-1\n"
                                "1,-1,22,22,4,4,0.1,-1,-1,-1\n")
            args = SimpleNamespace(threshold=.2, base_keep_score=.1, overlap_iou=.3,
                                   max_additions=100, score=.75)
            info = {"frame_numbering": "split", "source_frame_start": 4,
                    "source_frame_stop": 4, "width": 32, "height": 32}
            result = merge_sequence(base, proposal, root / "out", info, args)
            merged = (root / "out/det.txt").read_bytes()
            self.assertEqual(merged[:len(original)], original)
            self.assertEqual(result["added"], 1)
            self.assertEqual(result["overlap_original"], 1)
            self.assertEqual(result["overlap_added"], 1)
            self.assertEqual(result["below_threshold"], 1)
            self.assertEqual(parse_rows(merged, "test")[-1][1][6], .75)
            added = json.loads((root / "out/added.jsonl").read_text())
            self.assertEqual(added["raw_score"], .87654321)

    def test_merge_rejects_mismatched_frame_numbering_and_bounds(self):
        row = [(1, [2, -1, 1, 1, 4, 4, .5])]
        with self.assertRaises(ValueError):
            select_additions([], row, 32, 32, 1, 1, .2, .1, .3, 100)
        row = [(1, [1, -1, 30, 30, 4, 4, .5])]
        with self.assertRaises(ValueError):
            select_additions([], row, 32, 32, 1, 1, .2, .1, .3, 100)

    def test_export_uses_only_causal_split_history_and_reindexes_dev(self):
        class RecordingModel:
            def __init__(self):
                self.history = []

            def __call__(self, batch):
                self.history.append(batch[0, :, :, :8, :12].mean((1, 2, 3)).tolist())
                heat = torch.full((len(batch), 1, 4, 4), -20.0)
                heat[:, :, 1, 1] = 3
                return {"heatmap": heat, "offset": torch.full((len(batch), 2, 4, 4), .5),
                        "log_wh": torch.full((len(batch), 2, 4, 4), math.log(4))}

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sequence = root / "UAVSwarm-01"
            (sequence / "img1").mkdir(parents=True)
            (sequence / "seqinfo.ini").write_text("[Sequence]\nseqLength=6\nimWidth=12\nimHeight=8\nimDir=img1\nimExt=.png\n")
            for frame in range(1, 7):
                cv2.imwrite(str(sequence / "img1" / f"{frame:06d}.png"), np.full((8, 12, 3), frame * 20, np.uint8))
            args = SimpleNamespace(split="dev", tile_size=16, overlap=4, history_gap=1,
                                   frame_numbering="split", batch_size=2, threshold=.05,
                                   topk_per_tile=128, min_box_size=2, max_box_size=0,
                                   nms_iou=.5, topk=100)
            model = RecordingModel()
            result = export_sequence(model, sequence, root / "out", args, torch.device("cpu"))
            np.testing.assert_allclose(np.asarray(model.history) * 255, [[100, 100, 100], [120, 100, 100]], atol=1e-4)
            rows = parse_rows((root / "out/det.txt").read_bytes(), "export")
            self.assertEqual([row[0] for _, row in rows], [1, 2])
            self.assertEqual(result["source_frame_start"], 5)
            records = [json.loads(line) for line in (root / "out/frame_stats.jsonl").read_text().splitlines()]
            self.assertEqual(records[1]["history_frames"], [6, 5, 5])


if __name__ == "__main__":
    unittest.main()
