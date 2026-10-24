import copy
import unittest
import torch
from torch.nn import functional as F
from feedback.prompt import PriorPrompt
from feedback.learning_gate import regression_checks


class CenteredContentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.raw = PriorPrompt(dim=8, patch=4, bins=32, max_ids=8,
            position_encoding=True, content_score='cosine', spatial_score='box_gaussian',
            backprojection='peak_coverage', projection_init_gain=.01)
        self.detail = copy.deepcopy(self.raw)
        self.detail.value_content = 'patch_detail'
        self.boxes = torch.tensor([[1., 2., 4., 5.], [8., 9., 5., 4.]])
        self.ids = torch.tensor([1, 2])

    def tokens(self, m, image):
        return m.patch_embed(image).flatten(2).transpose(1, 2)[0]

    def test_equivalent_to_centered_bias_free_map_including_padding(self):
        x = F.pad(torch.randn(1, 3, 13, 17), (0, 3, 0, 3))
        m = self.detail
        mean = F.avg_pool2d(x, 4, 4).repeat_interleave(4, -2).repeat_interleave(4, -1)
        expected = F.linear(F.conv2d(x-mean, m.patch_embed.weight, stride=4)
            .flatten(2).transpose(1, 2)[0], m.value.weight)
        torch.testing.assert_close(m.patch_values(x, self.tokens(m, x)), expected, atol=2e-7, rtol=1e-5)
        # Removing affine constants must not accidentally reintroduce a template.
        with torch.no_grad():
            m.value.bias.add_(1.2)
            m.patch_embed.bias.add_(.7)
        torch.testing.assert_close(m.patch_values(x, self.tokens(m, x)), expected, atol=4e-7, rtol=1e-5)

    def test_piecewise_constant_color_is_zero_and_empty_prior_identity(self):
        # Exactly representable colors isolate structural cancellation from float32 mean rounding.
        x = torch.randint(-3, 4, (1, 3, 4, 5)).float().repeat_interleave(4, -2).repeat_interleave(4, -1)
        m = self.detail
        torch.testing.assert_close(m.patch_values(x, self.tokens(m, x)), torch.zeros(20, 8), atol=2e-7, rtol=0)
        torch.testing.assert_close(m(x, self.boxes, self.ids), x, atol=1e-7, rtol=0)
        self.assertTrue(torch.equal(m(x, torch.empty(0, 4), torch.empty(0, dtype=torch.long)), x))

    def test_raw_compatibility_same_lookup_and_nonzero_task_gradients(self):
        x = torch.randn(1, 3, 16, 20)
        m = self.raw
        t = self.tokens(m, x)
        a = m.attention(m.encode_query(self.boxes, self.ids, 16, 20), t, 16, 20, self.boxes)
        legacy = x + F.fold(m.to_patch(m.backproject(a, a @ m.value(t))).T[None], (16, 20), 4, stride=4)
        self.assertTrue(torch.equal(m(x, self.boxes, self.ids), legacy))
        d = self.detail
        td = self.tokens(d, x)
        ad = d.attention(d.encode_query(self.boxes, self.ids, 16, 20), td, 16, 20, self.boxes)
        self.assertTrue(torch.equal(a, ad))
        self.assertEqual(list(m.state_dict()), list(d.state_dict()))
        for k in m.state_dict():
            self.assertTrue(torch.equal(m.state_dict()[k], d.state_dict()[k]))
        d(x, self.boxes, self.ids).square().mean().backward()
        for name, param in d.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
        for name in ('key.weight', 'value.weight', 'patch_embed.weight', 'to_patch.weight', 'coordinate.0.weight'):
            self.assertGreater(dict(d.named_parameters())[name].grad.abs().max().item(), 0, name)
        self.assertEqual(d.value.bias.grad.abs().max().item(), 0)

    def test_real_yolox_loss_reaches_centered_content_with_frozen_detector(self):
        from feedback.model import build
        from feedback.train import configure_detector_training
        from feedback.learning_gate import state_hash
        m = build({'mode': 'pp', 'architecture': {'depth': .33, 'width': .25},
                   'prompt': {'dim': 8, 'patch': 4, 'bins': 32, 'max_ids': 8,
                       'projection_init_gain': .01, 'position_encoding': True,
                       'content_score': 'cosine', 'spatial_score': 'box_gaussian',
                       'backprojection': 'peak_coverage', 'value_content': 'patch_detail'}})
        configure_detector_training(m, True)
        before = state_hash(m.detector)
        x = torch.randn(2, 3, 64, 96)
        y = torch.tensor([[[0., 30., 25., 8., 6.]], [[0., 35., 25., 8., 6.]]])
        c = [{'boxes': torch.tensor([[25., 20., 8., 6.]]), 'ids': torch.tensor([1])}] * 2
        opt = torch.optim.SGD(m.feedback.parameters(), lr=1e-4)
        m(x, y, c)['total_loss'].backward()
        for name in ('key.weight', 'value.weight', 'patch_embed.weight', 'to_patch.weight', 'coordinate.0.weight'):
            grad = dict(m.feedback.named_parameters())[name].grad
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(grad.abs().max().item(), 0, name)
        self.assertTrue(all(p.grad is None for p in m.detector.parameters()))
        opt.step()
        self.assertEqual(before, state_hash(m.detector))

    def test_group_gate_rejects_low_signal_harm_or_wrong_history(self):
        gates = {'check_final_loss_over_off_max': 1.01, 'check_shifted_loss_margin_relative_min': 0.,
                 'delta_normalized_rms_min': 1e-5, 'delta_normalized_rms_max': .05,
                 'delta_normalized_maxabs_max': .5}
        good = {'correct': {'mean_loss': .99, 'deltas': [{'rms': 1e-4, 'max_abs': .01}]},
                'off': {'mean_loss': 1.}, 'shifted': {'mean_loss': 1.}}
        self.assertTrue(all(regression_checks(good, good, gates).values()))
        bad = copy.deepcopy(good); bad['correct']['deltas'][0]['rms'] = 9e-6
        self.assertFalse(regression_checks(bad, good, gates)['bounded_nontrivial_residual'])
        bad = copy.deepcopy(good); bad['correct']['mean_loss'] = 1.011
        self.assertFalse(regression_checks(good, bad, gates)['no_harm'])
        bad = copy.deepcopy(good); bad['shifted']['mean_loss'] = .98
        self.assertFalse(regression_checks(good, bad, gates)['correct_history'])


if __name__ == '__main__':
    unittest.main()
