"""One-step causal P3 memory with unmodified ByteTrack association."""
from types import SimpleNamespace
import numpy as np
import torch
from feedback.online import decode
from feedback.state import observe_detector_boxes
from .data import render_heat


def observed_history_xyxy(tracks):
    """Observation hook stores xywh, rasterizer consumes xyxy; never use KF boxes."""
    boxes=np.asarray([t.feedback_box for t in tracks if t.score>=.5],np.float32).reshape(-1,4)
    boxes[:,2:]+=boxes[:,:2]
    return boxes


class TraDeSOnline:
    def __init__(self, model, enabled=True, heat_shift=False):
        self.model=model.eval(); self.enabled=enabled
        self.heat_shift=heat_shift
        self.model.detector.head.decode_in_inference=False
        observe_detector_boxes()
        self.reset('unset')

    def reset(self, sequence):
        from yolox.tracker.byte_tracker import BYTETracker
        self.tracker=BYTETracker(SimpleNamespace(track_thresh=.6,track_buffer=30,
                                 match_thresh=.9,mot20=False),30)
        self.sequence=sequence;self.frame=0;self.previous=None
        self.history_boxes=np.empty((0,4),np.float32);self.geometry=None;self.counter=0

    @torch.no_grad()
    def step(self, image, sequence, frame, original_size, ratio):
        from yolox.tracker.basetrack import BaseTrack
        from yolox.utils import postprocess
        if sequence!=self.sequence or frame!=self.frame+1 or len(image)!=1 or ratio<=0:
            raise ValueError('Expected the next frame of the same sequence, batch one')
        geometry=(tuple(image.shape),tuple(original_size),float(ratio))
        if self.geometry is not None and self.geometry!=geometry:
            raise ValueError('Geometry changed; reset required')
        heat=render_heat(self.history_boxes*ratio,self.model.input_size)[None].to(image)
        if self.heat_shift:
            # Fixed spatial-support intervention; features and chronology unchanged.
            heat=heat.roll(heat.shape[-1]//2,dims=-1)
        result=self.model(image,previous_features=self.previous,heat=heat,enabled=self.enabled)
        head=self.model.detector.head
        raw=decode(result['output'].float().cpu(),head.hw,head.strides)
        output=postprocess(raw,1,.001,.7)[0]
        detections=np.empty((0,5),np.float64)
        if output is not None:
            boxes=output[:,:4].numpy().astype(np.float64)/ratio
            xywh=np.c_[boxes[:,:2],boxes[:,2:]-boxes[:,:2]].round(4)
            boxes=np.c_[xywh[:,:2],xywh[:,:2]+xywh[:,2:]]
            score=(output[:,4]*output[:,5]).numpy().astype(np.float64).round(6)
            detections=np.c_[boxes,score]
        counter=BaseTrack._count
        try:
            BaseTrack._count=self.counter
            tracks=self.tracker.update(detections.copy(),original_size,original_size)
            self.counter=BaseTrack._count
        finally:
            BaseTrack._count=counter
        # Use only current confirmed matched detector boxes, no KF-only extrapolation.
        self.history_boxes=observed_history_xyxy(tracks)
        self.previous=result['raw_feature'].detach()
        self.geometry=geometry;self.frame=frame
        return dict(detections=detections,
                    tracks=[(int(t.track_id),*t.feedback_box.tolist(),float(t.score)) for t in tracks],
                    feedback_used=result['cva'] is not None,history_frame=frame-1)
