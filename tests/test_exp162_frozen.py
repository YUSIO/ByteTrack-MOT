"""Frozen detector regressions using the real YOLOX loss/head at small width."""
import copy
import unittest

import torch

from feedback.weak import CausalState, WeakDetector
from feedback.weak_history import assert_frozen_detector, make_optimizer, train_batch


class FrozenDetector(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(162)
        self.model = WeakDetector(architecture={'depth': .33, 'width': .125})
        self.image = torch.randn(2, 3, 64, 96)
        self.heat = torch.rand(2, 1, 8, 12, requires_grad=True)
        self.targets = torch.tensor([[[0., 32., 32., 16., 16.]],
                                     [[0., 48., 32., 20., 20.]]])

    def test_mode_changes_never_enable_detector_or_bn(self):
        for mode in (True, False, True, True):
            self.model.train(mode)
            assert_frozen_detector(self.model)
            self.assertEqual(self.model.adapter.training, mode)
            self.assertTrue(all(not p.requires_grad for p in self.model.detector.parameters()))

    def test_updates_preserve_initial_weights_buffers_and_off_output(self):
        m = self.model.eval()
        original = copy.deepcopy(m.detector.state_dict())
        initial_adapter = copy.deepcopy(m.adapter.state_dict())
        with torch.no_grad():
            raw_before = m.detector(self.image).clone()
        # Synthetic randomly initialized tiny model: use a visible probe step.
        # This is not the formal recipe. No decay: prove a loss-driven update.
        optimizer = make_optimizer(m, lr=.1, weight_decay=0.)
        gradients = []
        for _ in range(3):
            train_batch(m, optimizer, self.image, self.heat, self.targets)
            gradients.append([p.grad.norm().item() for p in m.adapter.parameters()])
            assert_frozen_detector(m)
            for name, value in m.detector.state_dict().items():
                self.assertTrue(torch.equal(value, original[name]), name)
        self.assertEqual(gradients[0][0], 0.)
        self.assertGreater(gradients[0][1], 0.)
        self.assertGreater(gradients[1][0], 0.)
        self.assertIsNone(self.heat.grad)
        for name, value in m.adapter.state_dict().items():
            self.assertFalse(torch.equal(value, initial_adapter[name]), name)
        m.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(raw_before, m(self.image, self.heat, enabled=False)))
            self.assertTrue(torch.equal(raw_before, m(self.image, torch.zeros_like(self.heat))))
            self.assertFalse(torch.equal(raw_before, m(self.image, self.heat)))
        self.assertTrue(all(id(p) in {id(x) for x in m.adapter.parameters()}
                            for g in optimizer.param_groups for p in g['params']))

    def test_detector_in_optimizer_rejected_before_update(self):
        wrong = torch.optim.SGD(self.model.parameters(), lr=1e-4)
        with self.assertRaisesRegex(ValueError, 'exactly the adapter'):
            train_batch(self.model, wrong, self.image, self.heat, self.targets)

    def test_unfrozen_parameter_rejected(self):
        next(self.model.detector.parameters()).requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'frozen'):
            make_optimizer(self.model)

    def test_loss_exception_restores_eval_head(self):
        m = self.model.train()
        original = m.detector.head.forward
        def fail(*args, **kwargs):
            raise RuntimeError('intentional test')
        m.detector.head.forward = fail
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            m(self.image, self.heat, targets=self.targets)
        m.detector.head.forward = original
        assert_frozen_detector(m)

    def test_removed_finetuning_arm_rejected(self):
        with self.assertRaises(ValueError):
            CausalState('B1')


if __name__ == '__main__':
    unittest.main()
