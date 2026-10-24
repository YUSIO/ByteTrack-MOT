"""Exp162: one-frame unconsumed weak-detection feedback. No GT input API."""
import copy
import math
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .model import build
from .online import decode
from .state import observe_detector_boxes


def select_rows(detections, audit):
    detections = np.asarray(detections, dtype=np.float64).reshape(-1, 5)
    if len(audit) != len(detections):
        raise ValueError('Audit must cover every post-NMS row')
    t, u = [], []
    for i, (row, record) in enumerate(zip(detections, audit)):
        if record['det_row_id'] != i:
            raise ValueError('Unstable post-NMS row identity')
        if record['output_confirmed']:
            if record['consumed_by'] is None:
                raise ValueError('Confirmed output without a consumed detection')
            t.append(i)
        elif record['consumed_by'] is None and .1 < row[4] < .6:
            u.append(i)
    return t, u


def render(rows, ratio, hw=(100, 180), padding=(0., 0.)):
    """Original xyxy+score -> CPU float32 H; integer grid, floating centres.

    Radius is Euclidean 3 sigma; clipped only at grid boundaries. No score
    adjustment or position prediction. Rasterization is deliberately detached.
    """
    if ratio <= 0:
        raise ValueError('Resize ratio must be positive')
    heat = np.zeros(hw, np.float32)
    rows = np.asarray(rows, np.float64).reshape(-1, 5)
    if not np.isfinite(rows).all():
        raise ValueError('Nonfinite detections')
    for x1, y1, x2, y2, score in rows:
        if x2 <= x1 or y2 <= y1 or not 0 <= score <= 1:
            raise ValueError('Invalid detection geometry or score')
        cx = ((x1 + x2) * .5 * ratio + padding[0]) / 8.
        cy = ((y1 + y2) * .5 * ratio + padding[1]) / 8.
        lo_x, hi_x = max(0, math.ceil(cx-3)), min(hw[1]-1, math.floor(cx+3))
        lo_y, hi_y = max(0, math.ceil(cy-3)), min(hw[0]-1, math.floor(cy+3))
        if lo_x > hi_x or lo_y > hi_y:
            continue
        yy, xx = np.mgrid[lo_y:hi_y+1, lo_x:hi_x+1]
        d2 = (xx-cx)**2 + (yy-cy)**2
        peak = (score * np.exp(-.5*d2) * (d2 <= 9)).astype(np.float32)
        target = heat[lo_y:hi_y+1, lo_x:hi_x+1]
        np.maximum(target, peak, out=target)
    return torch.from_numpy(heat)[None]


class WeakAdapter(nn.Module):
    def __init__(self, channels=320):
        super().__init__()
        self.input = nn.Conv2d(1, 16, 3, padding=1, bias=False)
        self.output = nn.Conv2d(16, channels, 1, bias=False)
        nn.init.kaiming_normal_(self.input.weight)
        nn.init.zeros_(self.output.weight)

    def forward(self, heat):
        return self.output(torch.nn.functional.silu(self.input(heat.detach())))


class WeakDetector(nn.Module):
    def __init__(self, checkpoint=None, architecture=None):
        super().__init__()
        cfg = {'mode': 'none', 'architecture': architecture or {'depth': 1.33, 'width': 1.25}}
        self.detector = build(cfg, checkpoint).detector
        self.adapter = WeakAdapter(int(256 * cfg['architecture']['width']))

    def forward(self, images, heat=None, targets=None, enabled=True, return_residual=False):
        features = self.detector.backbone(images)
        if heat is None:
            heat = images.new_zeros((len(images), 1, *features[0].shape[-2:]))
        if heat.shape != (len(images), 1, *features[0].shape[-2:]):
            raise ValueError('Heatmap geometry mismatch')
        residual = self.adapter(heat.to(features[0])) if enabled else torch.zeros_like(features[0])
        features = (features[0] + residual, *features[1:])
        if self.training:
            if targets is None:
                raise ValueError('Current-frame detection targets required in train mode')
            values = self.detector.head(features, targets, images)
            result = dict(zip(('total_loss', 'iou_loss', 'conf_loss', 'cls_loss',
                               'l1_loss', 'num_fg'), values))
        else:
            result = self.detector.head(features)
        return (result, residual) if return_residual else result


class CausalState:
    def __init__(self, arm):
        if arm not in ('B0', 'B1', 'T', 'TU', 'TU_Uoff', 'TU_Ushift'):
            raise ValueError(arm)
        self.arm = arm
        self.reset('unset')

    def reset(self, sequence):
        self.sequence, self.frame = sequence, 0
        self.rows = np.empty((0, 5), np.float64)
        self.t, self.u = [], []

    def snapshot(self, sequence, next_frame, snapshot_sha):
        if sequence != self.sequence or next_frame != self.frame + 1:
            raise ValueError('History must be same-sequence and strictly previous-frame')
        return {'sequence': sequence, 'target_frame': next_frame, 'source_frame': self.frame,
                'snapshot_sha': snapshot_sha, 'arm': self.arm,
                'rows': self.rows.tolist(), 't': self.t.copy(), 'u': self.u.copy()}

    def heat(self, sequence, next_frame, ratio, hw):
        return heat_from_record(self.snapshot(sequence, next_frame, ''), ratio, hw)

    def advance(self, sequence, frame, rows, audit):
        if sequence != self.sequence or frame != self.frame + 1:
            raise ValueError('Noncausal or cross-sequence state update')
        t, u = select_rows(rows, audit)
        self.rows = np.asarray(rows, np.float64).reshape(-1, 5).copy()
        self.t, self.u, self.frame = t, u, frame


def heat_from_record(record, ratio, hw, expected_sequence=None, expected_frame=None,
                     expected_snapshot=None):
    if record['source_frame'] != record['target_frame'] - 1:
        raise ValueError('Future/stale history record')
    for field, expected in [('sequence', expected_sequence), ('target_frame', expected_frame),
                            ('snapshot_sha', expected_snapshot)]:
        if expected is not None and record[field] != expected:
            raise ValueError('History provenance mismatch: ' + field)
    rows = np.asarray(record['rows'], np.float64).reshape(-1, 5)
    if record['source_frame'] == 0 and (len(rows) or record['t'] or record['u']):
        raise ValueError('First frame cannot have history')
    if record['arm'] in ('B0', 'B1'):
        return torch.zeros((1, *hw))
    t = render(rows[record['t']], ratio, hw)
    if record['arm'] in ('T', 'TU_Uoff'):
        return t
    if record['arm'] not in ('TU', 'TU_Ushift'):
        raise ValueError('Unknown arm')
    u = render(rows[record['u']], ratio, hw)
    if record['arm'] == 'TU_Ushift':
        u = torch.roll(u, shifts=hw[1]//2, dims=-1)
    return torch.maximum(t, u)


class DetectionTracker:
    """ByteTrack row bookkeeping with isolated ID counter for paired runners."""
    def __init__(self, audit=True, tracker_class=None):
        from yolox.tracker.byte_tracker import BYTETracker
        observe_detector_boxes()
        cls = tracker_class or BYTETracker
        self.tracker = cls(SimpleNamespace(track_thresh=.6, track_buffer=30,
                          match_thresh=.9, mot20=False, audit_detections=audit), 30)
        self.counter = 0
        self.audit = audit

    def update(self, detections, shape):
        from yolox.tracker.basetrack import BaseTrack
        old = BaseTrack._count
        BaseTrack._count = self.counter
        try:
            tracks = self.tracker.update(detections.copy(), shape, shape)
            self.counter = BaseTrack._count
        finally:
            BaseTrack._count = old
        rows = [(int(t.track_id), *t.feedback_box.tolist(), float(t.score)) for t in tracks]
        return rows, copy.deepcopy(self.tracker.last_detection_audit) if self.audit else None


def postprocess_rows(raw, head, ratio):
    from yolox.utils import postprocess
    output = postprocess(decode(raw.float().cpu(), head.hw, head.strides), 1, .001, .7)[0]
    if output is None:
        return np.empty((0, 5), np.float64)
    boxes = output[:, :4].numpy().astype(np.float64) / ratio
    xywh = np.c_[boxes[:, :2], boxes[:, 2:] - boxes[:, :2]].round(4)
    boxes = np.c_[xywh[:, :2], xywh[:, :2] + xywh[:, 2:]]
    score = (output[:, 4] * output[:, 5]).numpy().astype(np.float64).round(6)
    return np.c_[boxes, score]


class WeakOnline:
    def __init__(self, model, arm, snapshot_sha):
        self.model = model.eval()
        self.model.detector.head.decode_in_inference = False
        self.state = CausalState(arm)
        self.snapshot_sha = snapshot_sha

    def reset(self, sequence):
        self.state.reset(sequence)
        self.tracker = DetectionTracker()
        self.geometry = None

    @torch.no_grad()
    def step(self, image, sequence, frame, shape, ratio):
        geometry = (tuple(image.shape), tuple(shape), float(ratio))
        if self.geometry is not None and geometry != self.geometry:
            raise ValueError('Geometry changed within sequence')
        self.geometry = geometry
        if image.shape[0] != 1:
            raise ValueError('Online history requires batch one')
        history = self.state.snapshot(sequence, frame, self.snapshot_sha)
        heat = heat_from_record(history, ratio, (image.shape[-2]//8, image.shape[-1]//8))
        raw = self.model(image, heat[None].to(image), enabled=self.state.arm != 'B0')
        dets = postprocess_rows(raw, self.model.detector.head, ratio)
        tracks, audit = self.tracker.update(dets, shape)
        self.state.advance(sequence, frame, dets, audit)
        return {'history': history, 'detections': dets, 'tracks': tracks, 'audit': audit}
