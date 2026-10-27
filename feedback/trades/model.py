"""TraDeS CVA/MFW migration to frozen YOLOX P3; no detector optimization."""
from types import SimpleNamespace
import torch
from torch import nn
from feedback.model import build
from .vendor.base_model import BaseModel


class TraDeSAdapter(nn.Module):
    def __init__(self, channels=320, feature_size=(100, 180)):
        super().__init__()
        h, w = feature_size
        if min(h, w) < 4 or h % 2 or w % 2:
            raise ValueError('CVA requires an even P3 grid of at least 4x4')
        self.feature_size = tuple(feature_size)
        self.input = nn.Conv2d(channels, 64, 1, bias=False)
        opt = SimpleNamespace(head_kernel=3, trades=True, output_h=h, output_w=w,
                              deform_kernel_size=3, device='cpu', clip_len=2)
        self.core = BaseModel({}, {}, 1, 64, opt)
        self.core.opt = opt
        self.output = nn.Conv2d(64, channels, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, current, previous=None, heat=None, enabled=True):
        if tuple(current.shape[-2:]) != self.feature_size:
            raise ValueError('P3 geometry differs from the configured grid')
        if current.dtype != torch.float32:
            raise ValueError('Initial migration is FP32 only; AMP has not been validated')
        zero = self.output.weight.sum() * 0
        if not enabled or previous is None or heat is None:
            return current, None, zero
        if previous.shape != current.shape or heat.shape != (len(current), 1, *self.feature_size):
            raise ValueError('Historical feature/heat geometry mismatch')
        if not torch.isfinite(heat).all() or (heat < 0).any() or (heat > 1).any():
            raise ValueError('History heat must be finite in [0,1]')
        # Only active histories enter BN/CVA; an empty member is an exact bypass.
        active = heat.detach().flatten(1).amax(1) > 0
        if not active.any():
            return current, None, zero
        cur = self.input(current[active].detach())
        prev = self.input(previous[active].detach())
        self.core.opt.device = current.device
        outputs = self.core.CVA_MFW([cur, prev], [prev * heat[active].detach()])
        delta = self.output(outputs[0][0] - cur)
        result = current.clone()
        result[active] = current[active] + delta
        cva = {'h': outputs[3][0], 'w': outputs[3][1], 'active': active,
               'offset': outputs[2]}
        return result, cva, delta


class FrozenTraDeS(nn.Module):
    def __init__(self, checkpoint=None, architecture=None, input_size=(800, 1440)):
        super().__init__()
        architecture = architecture or {'depth': 1.33, 'width': 1.25}
        self.architecture, self.input_size = architecture, tuple(input_size)
        if any(v % 32 for v in self.input_size):
            raise ValueError('YOLOX input dimensions must be multiples of 32')
        self.detector = build({'mode': 'none', 'architecture': architecture}, checkpoint).detector
        self.adapter = TraDeSAdapter(int(256*architecture['width']),
                                    tuple(v//8 for v in self.input_size))
        self.detector.requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.detector.eval()
        return self

    def optimizer_parameters(self):
        if any(p.requires_grad for p in self.detector.parameters()):
            raise RuntimeError('Detector parameters must remain frozen')
        return self.adapter.parameters()

    def forward(self, images, previous_images=None, previous_features=None,
                heat=None, targets=None, enabled=True):
        if tuple(images.shape[-2:]) != self.input_size:
            raise ValueError('Input geometry mismatch')
        if previous_images is not None and previous_features is not None:
            raise ValueError('Provide images or cached raw P3, not both')
        with torch.no_grad():
            features = self.detector.backbone(images)
            old = (self.detector.backbone(previous_images)[0] if previous_images is not None
                   else previous_features)
        raw = features[0]
        refined, cva, residual = self.adapter(raw, old, heat, enabled)
        features = (refined, *features[1:])
        head = self.detector.head
        if targets is not None:
            old_mode = head.training
            try:
                # The head loss dispatch is mutable, all child BN stay eval.
                head.training = True
                values = head(features, targets, images)
            finally:
                head.training = old_mode
            output = dict(zip(('total_loss','iou_loss','conf_loss','cls_loss','l1_loss','num_fg'), values))
        else:
            output = head(features)
        return {'output': output, 'raw_feature': raw.detach(), 'cva': cva, 'residual': residual}
