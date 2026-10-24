"""Freeze OFF SimOTA support for feedback training; keep current IoU soft labels.

Only training-time assignment changes. The upstream head still computes every
loss term and its normalization. This context is single-threaded and must not
span other forwards. It never installs persistent state on the model.
"""
from contextlib import contextmanager
import torch
from yolox.utils import bboxes_iou


@contextmanager
def matching_context(head, matches, capture=False):
    previous = head.__dict__.get('get_assignments')
    original = head.get_assignments

    @torch.no_grad()
    def assignment(*args, **kwargs):
        index, _, anchors, gt, classes, predictions = args[:6]
        if capture:
            result = original(*args, **kwargs)
            matches[index] = {
                'gt': gt.detach().clone(), 'classes': classes.detach().clone(),
                'anchors': anchors,
                'result': tuple(v.detach().clone() if torch.is_tensor(v) else v for v in result),
            }
            return result
        item = matches[index]
        if anchors != item['anchors'] or not torch.equal(gt, item['gt']) or not torch.equal(classes, item['classes']):
            raise ValueError('Fixed matching requires the same image ordering and GT')
        matched_classes, mask, _, gt_indices, count = item['result']
        # Same XYWH IoU and detached quality target as original YOLOX matching.
        # Only support/GT matching/count are fixed, not prediction quality.
        ious = bboxes_iou(gt[gt_indices], predictions[mask], False).diag()
        return matched_classes.clone(), mask.clone(), ious, gt_indices.clone(), count

    head.get_assignments = assignment
    try:
        yield matches
    finally:
        if previous is None:
            del head.get_assignments
        else:
            head.get_assignments = previous


def fixed_matching_loss(model, images, targets, contexts):
    if not model.training or any(p.requires_grad for p in model.detector.parameters()):
        raise ValueError('Fixed matching diagnostic requires frozen detector and training loss')
    if any(m.training for m in model.detector.modules() if isinstance(m, torch.nn.BatchNorm2d)):
        raise ValueError('Detector BatchNorm must be frozen')
    matches = {}
    with torch.no_grad(), matching_context(model.detector.head, matches, capture=True):
        model(images, targets, [None] * len(images))
    with matching_context(model.detector.head, matches):
        return model(images, targets, contexts)
