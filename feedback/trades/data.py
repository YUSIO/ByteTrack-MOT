"""Train-only current GT supervision and causal history; explicit migration labels."""
from collections import Counter
from types import SimpleNamespace
import math
import numpy as np
import torch
from torch.nn import functional as F
from feedback.data import read_train_sequences, read_image
from .vendor.image import gaussian_radius, draw_umich_gaussian
from .vendor.labels import get_cost_volume_label


def render_heat(boxes_xyxy, input_size, stride=8):
    """Original input-grid Gaussian/max rasterization, then stride-8 average pool.

    Unit amplitude as official inference. Rows must already be selected causally.
    """
    h, w = input_size
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    if not np.isfinite(boxes).all():
        raise ValueError('Nonfinite history boxes')
    heat = np.zeros((h, w), np.float32)
    for box in boxes:
        b = box.copy(); b[[0,2]] = b[[0,2]].clip(0,w-1); b[[1,3]] = b[[1,3]].clip(0,h-1)
        bw, bh = b[2:] - b[:2]
        if min(bw,bh) <= 0:
            continue
        radius = max(0,int(gaussian_radius((math.ceil(bh),math.ceil(bw)))))
        center = ((b[:2]+b[2:])/2).astype(np.int32)
        draw_umich_gaussian(heat,center,radius)
    return F.avg_pool2d(torch.from_numpy(heat)[None,None],stride,stride)[0]


def correspondence_targets(current, previous, ratio, input_size):
    """ID-matched GT centers. Keep original label algebra; exclude ambiguous query cells.

    Original dense labels overwrite if multiple current objects occupy one CVA
    cell. Exclude all such pairs instead of choosing by annotation order.
    """
    fh,fw = [v//8 for v in input_size]; ch,cw = fh//2,fw//2
    entries=[]
    for identity in sorted(set(current)&set(previous)):
        b=np.asarray(current[identity],np.float32).copy()*ratio
        b[2:]+=b[:2]
        b[[0,2]]=b[[0,2]].clip(0,input_size[1]-1)
        b[[1,3]]=b[[1,3]].clip(0,input_size[0]-1)
        old=np.asarray(previous[identity],np.float32)
        p=(old[:2]+old[2:]/2)*ratio
        q=(b[:2]+b[2:])/2
        if min(b[2:]-b[:2])<=0 or not (0<=p[0]<input_size[1] and 0<=p[1]<input_size[0]):
            continue
        cell=(int(q[1]/16),int(q[0]/16))
        entries.append((identity,b/8,p/8,cell))
    counts=Counter(e[3] for e in entries)
    kept=[e for e in entries if counts[e[3]]==1]
    n=len(kept)
    ret={'cat_down':np.zeros(n,np.int64),'mask_down':np.ones(n,np.float32),
         'h_ind_down':np.zeros(n,np.int64),'w_ind_down':np.zeros(n,np.int64),
         'h_hm_down':np.ones((ch*cw,ch),np.float32),
         'w_hm_down':np.ones((ch*cw,cw),np.float32)}
    owner=SimpleNamespace(opt=SimpleNamespace(output_h=fh,output_w=fw))
    for k,(_,bbox,prev,_) in enumerate(kept):
        get_cost_volume_label(owner,bbox,k,ret,prev)
    return {k:torch.from_numpy(v) for k,v in ret.items()}, {
        'common_valid_pairs':len(entries),'used_pairs':n,
        'collision_excluded_pairs':len(entries)-n}


def cva_loss(cva, targets):
    """Official CostVolumeLoss1D focal formula, evaluated with stable log-softmax.

    Each axis averages matched objects across the active batch; no inference
    temperature is added to the loss (matching official code).
    """
    if cva is None:
        raise ValueError('CVA supervision requires a nonempty active history')
    active=cva['active'].nonzero(as_tuple=False).flatten().tolist()
    total=cva['h'].sum()*0; count=0
    for local,batch in enumerate(active):
        target=targets[batch]; q=target['cat_down'].to(cva['h'].device)
        if not len(q):
            continue
        for axis in ('h','w'):
            logits=cva[axis][local,q]
            suppress=target[axis+'_hm_down'].to(logits)[q]
            logp=(logits*(1-suppress)).log_softmax(-1)
            idx=target[axis+'_ind_down'].to(logits.device)
            chosen=logp.gather(1,idx[:,None]).squeeze(1)
            total=total-((1-chosen.exp()).square()*chosen).sum()
        count+=len(q)
    return total/max(count,1)


class TrainPairs:
    def __init__(self, root, sequences, input_size=(800,1440)):
        self.data=read_train_sequences(root,sequences)
        self.input_size=tuple(input_size)
        self.index=[(name,t) for name,(_,info,_) in self.data.items()
                    for t in range(2,int(info['seqlength'])+1)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self,index):
        name,t=self.index[index];path,info,frames=self.data[name]
        image,ratio,shape=read_image(path,t,info,self.input_size)
        previous,old_ratio,old_shape=read_image(path,t-1,info,self.input_size)
        if ratio!=old_ratio or shape!=old_shape:
            raise ValueError('Temporal image geometry changed')
        old=np.asarray(list(frames[t-1].values()),np.float32).reshape(-1,4)*ratio
        old[:,2:]+=old[:,:2]
        heat=render_heat(old,self.input_size)
        boxes=np.asarray(list(frames[t].values()),np.float32).reshape(-1,4)
        targets=torch.zeros(max(1,len(boxes)),5)
        if len(boxes):
            targets[:len(boxes),1:3]=torch.from_numpy((boxes[:,:2]+boxes[:,2:]/2)*ratio)
            targets[:len(boxes),3:]=torch.from_numpy(boxes[:,2:]*ratio)
        cva,stats=correspondence_targets(frames[t],frames[t-1],ratio,self.input_size)
        return dict(image=image,previous=previous,heat=heat,targets=targets,cva=cva,
                    pair_stats=stats,sequence=name,frame=t,history_frame=t-1,
                    history_source='train_GT_teacher_forcing_no_disturbance')
