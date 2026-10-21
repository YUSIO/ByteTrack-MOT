"""Functional checks for motion direction, valid support, and control fairness."""

import io
import unittest

import torch

from prebirth.model import TemporalDetector, _shift_without_wrap


class PrebirthModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shift_is_translation_with_no_opposite_edge_wrap(self):
        image = torch.zeros(1, 1, 5, 7)
        image[0, 0, 1, 2] = 3
        image[0, 0, 3, 6] = 7
        shifted, valid = _shift_without_wrap(image, 2, -1)
        self.assertEqual(shifted[0, 0, 0, 4].item(), 3)
        self.assertEqual(shifted.sum().item(), 3)  # The rightmost source falls out.
        self.assertFalse(valid[..., :2].any())
        self.assertFalse(valid[..., -1, :].any())
        self.assertEqual(valid.sum().item(), 20)
        shifted, valid = _shift_without_wrap(image, 20, 0)
        self.assertEqual(shifted.sum().item(), 0)
        self.assertFalse(valid.any())

    def test_paths_use_consistent_displacement_at_both_ages(self):
        model = TemporalDetector(channels=8, temperature=0.05)
        current = torch.zeros(1, 2, 13, 13)
        previous, older = current.clone(), current.clone()
        current[0, 0, 6, 7] = 1
        previous[0, 0, 6, 5] = 1
        older[0, 0, 6, 3] = 1
        weights = model._path_weights(current, previous, older)
        correct = model.path_shifts.index((2, 0))
        self.assertGreater(weights[0, correct, 6, 7].item(), 0.999)
        aligned_previous, aligned_older = model._aligned_history(current, previous, older)
        self.assertGreater(aligned_previous[0, 0, 6, 7].item(), 0.999)
        self.assertGreater(aligned_older[0, 0, 6, 7].item(), 0.999)

    def test_boundary_masks_and_hovering_remain_finite(self):
        model = TemporalDetector(channels=8)
        image = torch.ones(1, 2, 3, 3)
        weights = model._path_weights(image, image, image)
        zero = model.path_shifts.index((0, 0))
        self.assertTrue(torch.isfinite(weights).all())
        # On a 3x3 grid the older +/-4 shifts have no valid support anywhere.
        torch.testing.assert_close(weights[:, zero], torch.ones_like(weights[:, zero]))
        previous, older = model._aligned_history(image, image, image)
        torch.testing.assert_close(previous, image)
        torch.testing.assert_close(older, image)

    def test_all_controls_have_same_parameters_and_expected_output(self):
        parameter_counts = []
        images = torch.rand(2, 3, 3, 35, 49)
        for mode in TemporalDetector.MODES:
            model = TemporalDetector(mode=mode, channels=8).eval()
            parameter_counts.append(sum(parameter.numel() for parameter in model.parameters()))
            with torch.no_grad():
                outputs = model(images)
            for key, channels in (("heatmap", 1), ("offset", 2), ("log_wh", 2)):
                self.assertEqual(outputs[key].shape, (2, channels, 9, 13))
                self.assertTrue(torch.isfinite(outputs[key]).all())
            self.assertLess(outputs["heatmap"].sigmoid().mean().item(), 0.02)
            self.assertGreater(outputs["log_wh"].exp().mean().item(), 20)
        self.assertEqual(len(set(parameter_counts)), 1)

    def test_single_does_not_read_history_even_in_training(self):
        torch.manual_seed(3)
        model = TemporalDetector(mode="single", channels=8).train()
        images = torch.rand(2, 3, 3, 24, 32, requires_grad=True)
        original = model(images)
        changed = images.detach().clone()
        changed[:, 1:] = torch.randn_like(changed[:, 1:]) * 100
        altered = model(changed)
        for key in original:
            torch.testing.assert_close(original[key], altered[key], rtol=0, atol=0)
        original["heatmap"].sum().backward()
        self.assertEqual(images.grad[:, 1:].abs().sum().item(), 0)
        self.assertGreater(images.grad[:, 0].abs().sum().item(), 0)

    def test_temporal_gradient_and_checkpoint_round_trip(self):
        torch.manual_seed(7)
        model = TemporalDetector(mode="aligned", channels=8, path_step=1).eval()
        images = torch.rand(1, 3, 3, 24, 28, requires_grad=True)
        output = model(images)
        sum(value.square().mean() for value in output.values()).backward()
        self.assertTrue(torch.isfinite(images.grad).all())
        for frame in range(3):
            self.assertGreater(images.grad[:, frame].abs().sum().item(), 0)
        checkpoint = io.BytesIO()
        torch.save({"model": model.state_dict(), "model_config": model.model_config}, checkpoint)
        checkpoint.seek(0)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        restored = TemporalDetector(**saved["model_config"]).eval()
        restored.load_state_dict(saved["model"])
        with torch.no_grad():
            for key, actual in restored(images.detach()).items():
                torch.testing.assert_close(actual, output[key])
            # Processing an unrelated window does not mutate a recurrent state.
            restored(torch.rand_like(images))
            for key, actual in restored(images.detach()).items():
                torch.testing.assert_close(actual, output[key])

    def test_invalid_inputs_fail_explicitly(self):
        model = TemporalDetector(channels=8)
        with torch.no_grad():
            self.assertEqual(model(torch.rand(1, 3, 3, 4, 4))["heatmap"].shape, (1, 1, 1, 1))
        with self.assertRaises(ValueError):
            model(torch.rand(1, 4, 3, 16, 16))
        with self.assertRaises(TypeError):
            model(torch.zeros(1, 3, 3, 16, 16, dtype=torch.uint8))
        with self.assertRaises(ValueError):
            TemporalDetector(mode="offline")
        with self.assertRaises(ValueError):
            TemporalDetector(temperature=0)


if __name__ == "__main__":
    unittest.main()
