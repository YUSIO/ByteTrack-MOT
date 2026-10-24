"""Batch-one online execution. No annotations are accepted by this interface."""
from types import SimpleNamespace
import numpy as np
import torch
from .state import History, observe_detector_boxes


def decode(raw, hw, strides):
    grids, scales = [], []
    for (h, w), stride in zip(hw, strides):
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        grids.append(torch.stack((x, y), -1).view(1, -1, 2))
        scales.append(torch.full((1, h*w, 1), float(stride)))
    result = raw.clone()
    stride = torch.cat(scales, 1).to(raw)
    result[..., :2] = (raw[..., :2] + torch.cat(grids, 1).to(raw)) * stride
    result[..., 2:4] = raw[..., 2:4].exp() * stride
    return result


class OnlineFeedback:
    def __init__(self, model, config, predictor=None, frame_rate=30):
        self.model, self.config, self.predictor = model.eval(), config, predictor
        if model.mode == 'eec' and predictor is None:
            raise ValueError('EEC requires an explicit PSMP predictor')
        if predictor is not None:
            predictor.eval().requires_grad_(False)
        model.detector.head.decode_in_inference = False
        self.frame_rate = frame_rate
        observe_detector_boxes()
        self.reset()

    def reset(self):
        from yolox.tracker.byte_tracker import BYTETracker
        from yolox.tracker.basetrack import BaseTrack
        # ByteTrack owns a process-global ID counter: runners must not interleave.
        BaseTrack._count = 0
        self.tracker = BYTETracker(SimpleNamespace(**self.config['tracker']), self.frame_rate)
        self.history = History(self.config.get('window', 8))
        self.prompts, self.previous, self.geometry = {}, None, None

    @torch.no_grad()
    def step(self, image, original_size, ratio):
        from yolox.utils import postprocess
        if image.shape[0] != 1 or ratio <= 0:
            raise ValueError('Online inference requires one image and a positive resize ratio')
        geometry = (tuple(image.shape), tuple(original_size), float(ratio))
        if self.geometry is not None and geometry != self.geometry:
            raise ValueError('Geometry changed within sequence; reset state explicitly')
        self.geometry = geometry
        context, used, reliable_count = None, False, 0
        if self.model.mode == 'pp' and self.prompts:
            ids = sorted(self.prompts)
            context = {'boxes': image.new_tensor(np.stack([self.prompts[i] for i in ids])) * ratio,
                       'ids': torch.tensor(ids, device=image.device)}
            used = True
        elif self.model.mode == 'eec' and self.previous is not None:
            positions, ids = self.history.reliable()
            reliable_count = len(ids)
            if ids:
                # PSMP and its uncertainty use detector-input pixels, then /8 and /64 in EEC.
                prior = self.predictor(image.new_tensor(positions)[None] * ratio)['prior']
                context = {'previous': self.previous, 'prior': prior}
                used = True
        raw, feature = self.model(image, contexts=[context], return_raw=True)
        head = self.model.detector.head
        output = postprocess(decode(raw.float().cpu(), head.hw, head.strides), 1,
                             self.config.get('conf', .001), self.config.get('nms', .7))[0]
        detections = np.empty((0, 5), np.float64)
        if output is not None:
            boxes = output[:, :4].numpy().astype(np.float64) / ratio
            # Match Exp141's stored det.txt precision before ByteTrack consumes it.
            xywh = np.c_[boxes[:, :2], boxes[:, 2:] - boxes[:, :2]].round(4)
            boxes = np.c_[xywh[:, :2], xywh[:, :2] + xywh[:, 2:]]
            score = (output[:, 4] * output[:, 5]).numpy().astype(np.float64).round(6)
            detections = np.c_[boxes, score]
        tracks = self.tracker.update(detections.copy(), original_size, original_size)
        frame = self.tracker.frame_id
        observed = {t.track_id: t.feedback_box.copy() for t in self.tracker.tracked_stracks
                    if t.frame_id == frame and hasattr(t, 'feedback_box')}
        self.history.append(frame, observed)
        self.prompts = {t.track_id: t.feedback_box.copy() for t in tracks}
        self.previous = feature.detach() if self.model.mode == 'eec' else None
        rows = [(t.track_id, *t.feedback_box.tolist(), float(t.score)) for t in tracks]
        return {'frame': frame, 'detections': detections, 'tracks': rows,
                'feedback_used': used, 'reliable_count': reliable_count,
                'history_last_frame_used': frame - 1}
