"""Shared YOLOX detector wrapper. Losses/head/backbone stay in upstream YOLOX."""
import torch
from torch import nn
from .prompt import PriorPrompt


class FeedbackDetector(nn.Module):
    def __init__(self, detector, mode='none', prompt_config=None, eec_config=None):
        super().__init__()
        if mode not in ('none', 'pp', 'eec'):
            raise ValueError(mode)
        self.detector, self.mode = detector, mode
        if mode == 'pp':
            self.feedback = PriorPrompt(**(prompt_config or {}))
        elif mode == 'eec':
            from .eec import EECActivation
            self.feedback = EECActivation(**(eec_config or {}))

    def features(self, images, contexts=None, previous_images=None):
        contexts = contexts if contexts is not None else [None] * len(images)
        if len(contexts) != len(images):
            raise ValueError('One context per image required')
        if self.mode == 'pp':
            refined = []
            for image, context in zip(images.split(1), contexts):
                refined.append(image if context is None else self.feedback(
                    image, context['boxes'].to(image), context['ids'].to(image.device)))
            images = torch.cat(refined)
        current = self.detector.backbone(images)
        raw = current[0]  # cache unmodified current stride-8 feature
        if self.mode == 'eec':
            previous = None
            if previous_images is not None:
                previous = self.detector.backbone(previous_images)[0]
            enhanced = []
            for i, context in enumerate(contexts):
                if context is None:
                    enhanced.append(raw[i:i+1])
                    continue
                old = previous[i:i+1] if previous is not None else context['previous']
                enhanced.append(self.feedback(raw[i:i+1], old, context['prior']))
            current = (torch.cat(enhanced), *current[1:])
        return current, raw, images

    def forward(self, images, targets=None, contexts=None, previous_images=None,
                return_raw=False):
        features, raw, refined = self.features(images, contexts, previous_images)
        if self.training:
            if targets is None:
                raise ValueError('Training requires current-frame detection targets')
            values = self.detector.head(features, targets, refined)
            result = dict(zip(('total_loss', 'iou_loss', 'conf_loss', 'cls_loss',
                               'l1_loss', 'num_fg'), values))
        else:
            result = self.detector.head(features)
        return (result, raw) if return_raw else result


def build(config, checkpoint=None):
    from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
    architecture = config.get('architecture', {'depth': 1.33, 'width': 1.25})
    detector = YOLOX(YOLOPAFPN(**architecture), YOLOXHead(1, width=architecture['width']))
    # Match YOLOX Exp.get_model() BatchNorm configuration.
    for layer in detector.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eps, layer.momentum = 1e-3, 0.03
    detector.head.initialize_biases(1e-2)
    wrapped = FeedbackDetector(detector, config['mode'], config.get('prompt'), config.get('eec'))
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if 'feedback_config' in state:
            if state['feedback_config'] != config:
                raise ValueError('Checkpoint configuration differs from requested configuration')
            wrapped.load_state_dict(state['model'], strict=True)
        else:
            detector.load_state_dict(state['model'], strict=True)
    return wrapped
