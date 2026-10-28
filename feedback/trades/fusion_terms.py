"""Read-only interventions on one native TraDeS forward; attention is shared and fixed."""
import torch

MODES = ('off', 'full', 'no_current', 'no_visual', 'visual_only')


@torch.no_grad()
def frozen_fusion_terms(adapter, current, previous, heat):
    if adapter.training:
        raise ValueError('Fusion intervention requires eval mode')
    captured = {'input': [], 'prop': [], 'cur_logits': [], 'prev_logits': []}
    handles = []
    def record(key):
        def hook(module, args, result):
            captured[key].append(result.detach())
        return hook
    try:
        for module, key in ((adapter.input, 'input'), (adapter.core.dcn1_1, 'prop'),
                            (adapter.core.attention_cur, 'cur_logits'),
                            (adapter.core.attention_prev, 'prev_logits')):
            handles.append(module.register_forward_hook(record(key)))
        # Preserve native arithmetic/order for byte-identical baseline detections.
        native, cva, native_delta = adapter(current, previous, heat)
    finally:
        for handle in handles:
            handle.remove()
    refined = {'off': current, 'full': native}
    if cva is None:
        if any(captured.values()):
            raise RuntimeError('Bypass unexpectedly evaluated adapter internals')
        for mode in MODES[2:]:
            refined[mode] = current
        return refined, dict(active=False, attention_evaluations=0, decomposition_max_abs_error=0.,
                             native_reconstruction_max_abs_error=0., visual_rms=0., bias_rms=0., current_rms=0.,
                             attention_min=0., attention_max=0., attention_mean=0.), None
    if [len(captured[k]) for k in captured] != [2, 1, 1, 1]:
        raise RuntimeError('Expected exactly one native CVA/MFW/attention forward')
    c = captured['input'][0]; p = captured['prop'][0]
    weights = torch.cat((captured['cur_logits'][0], captured['prev_logits'][0]), dim=1).softmax(1)
    a = weights[:, 1:2].detach().clone()
    fixed_a = a.clone()
    b = adapter.core.dcn1_1.bias[None, :, None, None]
    v = p - b
    visual = adapter.output(a * v)
    bias = adapter.output(a * b)
    negative_current = adapter.output(-a * c)
    residuals = dict(no_current=adapter.output(a * p), no_visual=adapter.output(a * (b - c)),
                     visual_only=visual)
    for mode, residual in residuals.items():
        value = current.clone()
        value[cva['active']] = current[cva['active']] + residual
        refined[mode] = value
    if not torch.equal(a, fixed_a):
        raise RuntimeError('Attention changed between interventions')
    native_rebuilt = adapter.output(weights[:, :1] * c + a * p - c)
    native_error = (native_delta - native_rebuilt).abs().max()
    decomposition_error = (native_delta - (visual + bias + negative_current)).abs().max()
    scale = max(float(native_delta.abs().max()), 1.)
    if float(native_error) > 2e-5 * scale or float(decomposition_error) > 2e-5 * scale:
        raise RuntimeError('Fusion algebra mismatch')
    def rms(x): return float(x.square().mean().sqrt())
    audit = dict(active=True, attention_evaluations=1, decomposition_max_abs_error=float(decomposition_error),
                 native_reconstruction_max_abs_error=float(native_error), visual_rms=rms(visual), bias_rms=rms(bias),
                 current_rms=rms(negative_current), attention_min=float(a.min()), attention_max=float(a.max()),
                 attention_mean=float(a.mean()))
    return refined, audit, a
