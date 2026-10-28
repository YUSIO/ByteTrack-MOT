"""Frozen dense CVA reliability and recipient gate; inference never uses GT."""
import math
import torch
from torch.nn import functional as F


def reliability(cost, displacement, valid_hw):
    """Score native motion, without replacing it with a new correspondence.

    cost [1,N,N], native displacement [1,2,H,W] in P3 cells.
    Forward/backward distributions exclude pure-padding candidates only.
    """
    h,w=displacement.shape[-2:];hc,wc=h//2,w//2;n=hc*wc
    assert cost.shape==(1,n,n)
    rh,rw=valid_hw
    yy,xx=torch.meshgrid(torch.arange(hc,device=cost.device),torch.arange(wc,device=cost.device),indexing='ij')
    valid=((yy*16<rh)&(xx*16<rw)).flatten()
    logits=cost[0]*5
    pf=logits.masked_fill(~valid[None],float('-inf')).softmax(1)
    pb=logits.masked_fill(~valid[:,None],float('-inf')).softmax(0)
    # For each source p sum backward probability in the query's 3x3 neighborhood.
    back=F.avg_pool2d(pb.T.reshape(n,1,hc,wc),3,stride=1,padding=1,divisor_override=1).reshape(n,n).T
    cycle=(pf*back).sum(1).clamp(0,1)
    strict=(pf*pb).sum(1).clamp(0,1)
    d=displacement[0,:,::2,::2]/2
    px,py=xx+d[0],yy+d[1]
    grid=torch.stack((px*2/max(wc-1,1)-1,py*2/max(hc-1,1)-1),-1).reshape(n,1,1,2)
    endpoint=F.grid_sample(pf.reshape(n,1,hc,wc),grid,align_corners=True)[:,0,0,0]
    endpoint=(endpoint/pf.amax(1).clamp_min(1e-30)).clamp(0,1)
    valid_endpoint=(px>=0)&(py>=0)&(px<math.ceil(rw/16))&(py<math.ceil(rh/16))
    score=(cycle*endpoint*valid*valid_endpoint.flatten()).reshape(1,1,hc,wc)
    r=F.interpolate(score,size=(h,w),mode='nearest')
    p3_valid=(torch.arange(h,device=r.device)[:,None]*8<rh)&(torch.arange(w,device=r.device)[None,:]*8<rw)
    r=r*p3_valid[None,None]
    return r,dict(cycle_mean=float(cycle[valid].mean()),strict_cycle_mean=float(strict[valid].mean()),
                  endpoint_mean=float(endpoint[valid].mean()),r_mean=float(r[0,0][p3_valid].mean()))


def recipient_mask(detections,ratio,shape,valid_hw,device):
    """Max score times fractional box coverage; boxes are original xyxy+score."""
    h,w=shape
    result=torch.zeros((1,1,h,w),device=device)
    if len(detections):
        det=torch.as_tensor(detections,device=device,dtype=torch.float32)
        boxes=det[:,:4]*ratio/8
        x=torch.arange(w,device=device)[None]
        y=torch.arange(h,device=device)[None]
        wx=(torch.minimum(x+1,boxes[:,2:3])-torch.maximum(x,boxes[:,0:1])).clamp(0,1)
        wy=(torch.minimum(y+1,boxes[:,3:4])-torch.maximum(y,boxes[:,1:2])).clamp(0,1)
        result[0,0]=(wy[:,:,None]*wx[:,None,:]*det[:,4,None,None]).amax(0)
    rh,rw=valid_hw
    valid=(torch.arange(h,device=device)[:,None]*8<rh)&(torch.arange(w,device=device)[None,:]*8<rw)
    return result*valid[None,None]


def geometry(adapter,raw,old,valid_hw):
    """Compute original CVA and A/D offsets once; independent of historical heat."""
    core=adapter.core;core.opt.device=raw.device
    cur,prev=adapter.input(raw),adapter.input(old)
    b,_,h,w=cur.shape;hc,wc=h//2,w//2
    ce=core.maxpool_stride2(core.embedconv(cur)).flatten(2).transpose(1,2)
    pe=core.maxpool_stride2(core.embedconv(prev)).flatten(2)
    cost=ce@pe
    volume=cost.view(b,hc*wc,hc,wc)
    ch,cw=volume.max(3).values,volume.max(2).values
    # Preserve the native sum(probability * offset_template) operation order.
    v=torch.as_tensor(core.v,device=raw.device)
    m=torch.as_tensor(core.m,device=raw.device)
    dh=((ch*5).softmax(2)*v).sum(2).reshape(b,1,hc,wc)
    dw=((cw*5).softmax(2)*m).sum(2).reshape(b,1,hc,wc)
    d=F.interpolate(torch.cat((dw,dh),1),scale_factor=2,mode='nearest')
    native=core.gamma(d,cur,prev,b,h,w)
    ah=F.conv2d(d[:,1:2],core.conv_offset_h.weight[:,:1],padding=1)
    aw=F.conv2d(d[:,:1],core.conv_offset_w.weight[:,:1],padding=1)
    cv=torch.stack((ah,aw),2).flatten(1,2)
    direct=d[:,[1,0]][:,None].expand(-1,9,-1,-1,-1).flatten(1,2)
    anchored=native-cv+direct
    r,stats=reliability(cost,d,valid_hw)
    return dict(cur=cur,prev=prev,displacement=d,A=native,D=anchored,reliability=r,stats=stats)


def residuals(adapter,geo,heat):
    core=adapter.core
    support=geo['prev']*heat
    mask=torch.ones((len(support),9,*support.shape[-2:]),device=support.device)
    pa=core.dcn1_1(support,geo['A'],mask)
    pd=core.dcn1_1(support,geo['D'],mask)
    attention=torch.cat((core.attention_cur(geo['cur']),core.attention_prev(pa)),1).softmax(1)[:,1:2]
    bias=core.dcn1_1.bias[None,:,None,None]
    a=adapter.output(attention*(pa-bias))
    d=adapter.output(attention*(pd-bias))
    return a,d,attention


def blend(raw,a,d,gate):
    return raw+a+gate*(d-a)
