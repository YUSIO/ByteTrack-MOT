"""CPU preflight for quality-only gradients and frozen BN state."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fctrack.verifier import SeedVerifier
from tools.train_weak_birth_quality import sample_weights, train_mode, bn_state


class QualityTests(unittest.TestCase):
    def test_weights_split_duplicates_and_balance_classes(self):
        rows = [dict(sequence='a', frame='1', owner='2', label='1'),
                dict(sequence='a', frame='1', owner='2', label='1'),
                dict(sequence='a', frame='2', owner='2', label='1'),
                dict(sequence='a', frame='1', owner='-1', label='0')]
        w = sample_weights(rows)
        np.testing.assert_allclose(w, [.125, .125, .25, .5])

    def test_quality_gradient_and_bn_invariant(self):
        torch.set_num_threads(1)
        torch.manual_seed(181)
        model = SeedVerifier()
        before = bn_state(model)
        train_mode(model)
        self.assertTrue(model.head[2].training)
        self.assertFalse(model.body[0][1].training)
        logits = model(torch.rand(2, 3, 64, 64), torch.zeros(2, 2))[:, 0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.]))
        loss.backward()
        self.assertGreater(model.body[0][0].weight.grad.abs().sum().item(), 0)
        self.assertEqual(model.head[-1].weight.grad[1:].abs().sum().item(), 0)
        self.assertGreater(model.head[-1].weight.grad[0].abs().sum().item(), 0)
        self.assertTrue(all(torch.equal(v, bn_state(model)[k]) for k, v in before.items()))


if __name__ == '__main__':
    unittest.main()
