import unittest
from unittest.mock import patch
import torch
from feedback.trades.model import TraDeSAdapter
from feedback.trades.fusion_terms import frozen_fusion_terms, MODES
from feedback.trades.preflight import tensor_hashes


class FusionTermsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(17)
        self.model = TraDeSAdapter(8, (4, 6)).eval()
        with torch.no_grad():
            self.model.output.weight.normal_(0, .1)
            self.model.core.dcn1_1.bias.normal_(0, .15)
        self.cur = torch.randn(2, 8, 4, 6)
        self.prev = torch.randn_like(self.cur)
        self.heat = torch.rand(2, 1, 4, 6)

    def test_native_full_is_bit_identical_and_all_states_frozen(self):
        before = tensor_hashes(self.model)
        with torch.no_grad(): expected = self.model(self.cur, self.prev, self.heat)[0]
        values, audit, a = frozen_fusion_terms(self.model, self.cur, self.prev, self.heat)
        self.assertTrue(torch.equal(values['full'], expected))
        self.assertTrue(torch.equal(values['off'], self.cur))
        self.assertEqual(a.shape, (2, 1, 4, 6))
        self.assertEqual(audit['attention_evaluations'], 1)
        self.assertEqual(tensor_hashes(self.model), before)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_variants_share_native_attention_and_exact_formulas(self):
        with patch.object(self.model.core.attention_prev, 'forward', wraps=self.model.core.attention_prev.forward) as prev_call:
            values, audit, a = frozen_fusion_terms(self.model, self.cur, self.prev, self.heat)
            self.assertEqual(prev_call.call_count, 1)
        with torch.no_grad():
            c = self.model.input(self.cur); pprev = self.model.input(self.prev)
            capture = []
            h = self.model.core.dcn1_1.register_forward_hook(lambda m, args, output: capture.append(output))
            self.model(self.cur, self.prev, self.heat); h.remove()
            p = capture[0]; b = self.model.core.dcn1_1.bias[None, :, None, None]
            expressions = {'no_current': a*p, 'no_visual': a*(b-c), 'visual_only': a*(p-b)}
            for key, value in expressions.items():
                self.assertTrue(torch.equal(values[key], self.cur + self.model.output(value)), key)
        self.assertLess(audit['decomposition_max_abs_error'], 1e-5)

    def test_bias_contrast_equals_projected_bias_with_shared_attention(self):
        values, _, a = frozen_fusion_terms(self.model, self.cur, self.prev, self.heat)
        with torch.no_grad(): b = self.model.output(a*self.model.core.dcn1_1.bias[None,:,None,None])
        torch.testing.assert_close(values['no_current']-values['visual_only'], b, atol=5e-7, rtol=2e-5)

    def test_zero_support_visual_only_is_off_even_with_nonzero_bias(self):
        def zero(m, args): return args[0]*0, args[1], args[2]
        h = self.model.core.dcn1_1.register_forward_pre_hook(zero)
        try: values, _, _ = frozen_fusion_terms(self.model, self.cur, self.prev, self.heat)
        finally: h.remove()
        self.assertTrue(torch.equal(values['visual_only'], self.cur))
        self.assertFalse(torch.equal(values['no_current'], self.cur))
        self.assertFalse(torch.equal(values['full'], self.cur))

    def test_empty_and_missing_histories_all_exactly_bypass(self):
        for previous, heat in [(None, self.heat), (self.prev, torch.zeros_like(self.heat))]:
            values, audit, a = frozen_fusion_terms(self.model, self.cur, previous, heat)
            self.assertIsNone(a); self.assertFalse(audit['active'])
            for mode in MODES: self.assertTrue(torch.equal(values[mode], self.cur), mode)

    def test_mixed_batch_inactive_member_is_unchanged(self):
        self.heat[1].zero_()
        values, audit, a = frozen_fusion_terms(self.model, self.cur, self.prev, self.heat)
        self.assertEqual(a.shape[0], 1)
        for mode in MODES: self.assertTrue(torch.equal(values[mode][1], self.cur[1]), mode)

    def test_hook_cleanup_after_native_exception(self):
        with patch.object(self.model.core.dcn1_1, 'forward', side_effect=RuntimeError('injected')):
            with self.assertRaises(RuntimeError): frozen_fusion_terms(self.model, self.cur, self.prev, self.heat)
        for mod in [self.model.input,self.model.core.dcn1_1,self.model.core.attention_cur,self.model.core.attention_prev]:
            self.assertEqual(len(mod._forward_hooks),0)

    def test_train_mode_is_rejected(self):
        self.model.train()
        with self.assertRaises(ValueError): frozen_fusion_terms(self.model,self.cur,self.prev,self.heat)

if __name__ == '__main__': unittest.main()
