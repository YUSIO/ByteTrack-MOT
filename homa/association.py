"""Causal online appearance history and paper-form motion accumulation."""
from collections import deque
import numpy as np
import torch
from .model import tracklet_similarity
from yolox.tracker import matching


class HOMAAssociation:
    def __init__(self, model, arm, window=8, motion_window=8, kappa=.1, reduction='sum', high_threshold=.6):
        self.model,self.arm,self.window,self.motion_window=model,arm,window,motion_window
        self.kappa,self.reduction,self.high_threshold=kappa,reduction,high_threshold
        self.history=deque()
        self.motion={}
        self.current=None
        self.frame=0
        self.last_similarity=None

    def set_current(self, features, frame):
        self.current=features
        self.frame=frame
        while self.history and self.history[0][0]<frame-self.window+1:self.history.popleft()

    def motion_similarity(self, tracks, detections):
        d=np.array([[b.tlwh[0]+b.tlwh[2]/2,b.tlwh[1]+b.tlwh[3]/2,b.tlwh[2]/b.tlwh[3],b.tlwh[2]/b.tlwh[3]] for b in detections],dtype=float).reshape(-1,4)
        sim=np.zeros((len(tracks),len(detections)))
        for i,t in enumerate(tracks):
            h=self.motion.setdefault(t.track_id,deque())
            h.append((self.frame,np.r_[t.mean[:2],np.sqrt(np.diag(t.covariance)[:2])]))
            while h and h[0][0]<self.frame-self.motion_window+2:h.popleft()
            w=((np.stack([a[1] for a in h])[:,None]-d[None])**2).sum(2)
            distance=w.sum(0) if self.reduction=='sum' else w.mean(0)
            sim[i]=self.kappa/(self.kappa+distance)
        return sim

    @torch.inference_mode()
    def cost(self,tracks,detections):
        if self.arm=='mha_iou':return matching.iou_distance(tracks,detections)
        if self.arm=='mpa_iou': spatial=1-matching.iou_distance(tracks,detections)
        else: spatial=self.motion_similarity(tracks,detections)
        if self.arm=='m2da':return 1-spatial
        if not tracks or not detections:return np.ones((len(tracks),len(detections)))
        owners={t.track_id:i for i,t in enumerate(tracks)}
        items=[(f,tid,e) for f,entries in self.history for tid,e in entries if tid in owners]
        if not items:
            # No appearance-supported link, left unmatched; lifecycle is ByteTrack.
            return np.ones((len(tracks),len(detections)))
        device=next(self.model.parameters()).device
        history=torch.stack([e for _,_,e in items]).to(device)
        current=torch.stack([d.homa_feature for d in detections]).to(device)
        with torch.autocast(device.type,enabled=device.type=='cuda'):
            _,logits=self.model.logits(current,history)
        frames=torch.tensor([f for f,_,_ in items],device=device)
        indices=torch.tensor([owners[tid] for _,tid,_ in items],device=device)
        appearance=tracklet_similarity(logits,frames,indices,len(tracks)).cpu().numpy().T
        self.last_similarity=appearance*spatial
        # Eq.7 is a sum, so similarity may exceed 1 and cost may be negative.
        return 1-self.last_similarity

    def remember(self,tracks):
        entries=[]
        for t in tracks:
            if self.current is not None and getattr(t,'homa_feature_frame',None)==self.frame and t.score>self.high_threshold:
                entries.append((t.track_id,t.homa_feature.detach().cpu()))
        if entries:self.history.append((self.frame,entries))
        active={t.track_id for t in tracks}
        self.motion={k:v for k,v in self.motion.items() if k in active or (v and self.frame-v[-1][0]<=30)}
