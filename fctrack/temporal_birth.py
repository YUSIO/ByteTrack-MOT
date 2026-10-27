"""Causal three-frame crop construction and independent trainable birth classifier."""
from collections import OrderedDict
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from fctrack.verifier import SeedVerifier,seed_crop
from fctrack.tracker import iou_matrix


def adjacent_links(previous,current):
    """Frame-local full-cache indices. No GT or tracker outputs enter these links."""
    links=np.full(len(current),-1,np.int64)
    if not len(previous) or not len(current):return links
    a=np.asarray(current[:,:4],float);b=np.asarray(previous[:,:4],float)
    sa=np.sqrt(np.maximum(a[:,2]*a[:,3],1e-9));sb=np.sqrt(np.maximum(b[:,2]*b[:,3],1e-9))
    distance=np.linalg.norm((a[:,:2]+a[:,2:]/2)[:,None]-(b[:,:2]+b[:,2:]/2)[None],axis=2)/np.sqrt(sa[:,None]*sb[None])
    ratio=sa[:,None]/sb[None];ov=iou_matrix(a,b)
    valid=(distance<=2)&(ratio>=.5)&(ratio<=2)&(current[:,4,None]>=.001)&(previous[None,:,4]>=.001)&(a[:,2,None]>1)&(a[:,3,None]>1)&(b[None,:,2]>1)&(b[None,:,3]>1)
    edges=sorted(zip(*np.nonzero(valid)),key=lambda ij:(-ov[ij],distance[ij],ij[0],ij[1]))
    used=set()
    for i,j in edges:
        if links[i]<0 and j not in used:links[i]=j;used.add(j)
    return links


class ClipBuilder:
    def __init__(self,sequence_dir,cache):
        self.directory=Path(sequence_dir);self.frames={};self.links={};self.images=OrderedDict();self.ones={}
        for f in sorted(set(cache[:,0].astype(int))):
            r=cache[cache[:,0]==f];xyxy=np.column_stack((r[:,2:4],r[:,2:4]+r[:,4:6],r[:,6])).astype(np.float32)
            self.frames[f]=np.column_stack((xyxy[:,:2],xyxy[:,2:4]-xyxy[:,:2],xyxy[:,4])).astype(float)
        for f,current in self.frames.items():self.links[f]=adjacent_links(self.frames.get(f-1,np.empty((0,5))),current)
    def image(self,frame):
        if frame not in self.images:
            im=cv2.imread(str(self.directory/'img1'/f'{frame:06d}.jpg'));assert im is not None,(self.directory,frame)
            self.images[frame]=im
            if len(self.images)>6:self.images.popitem(last=False)
        return self.images[frame]
    def clip(self,frame,index):
        current=self.frames[frame][index];side=np.sqrt(current[2]*current[3]);centre=current[:2]+current[2:4]/2
        paths={frame:(current,True)};cursor=index;linked=True
        for f in [frame-1,frame-2]:
            if linked and f>=1:
                prev=self.links.get(f+1,np.empty(0,dtype=int))[cursor]
                if prev>=0:cursor=int(prev);paths[f]=(self.frames[f][cursor],True)
                else:linked=False
            else:linked=False
            if not linked:paths[f]=(current,False)
        images=[];metadata=[];width=self.image(frame).shape[1]
        for f in [frame-2,frame-1,frame]:
            if f<1:
                images.append(np.zeros((3,64,64),np.uint8));metadata.extend([0.]*8);continue
            box,has_link=paths[f];ss=np.sqrt(box[2]*box[3]);cc=box[:2]+box[2:4]/2;im=self.image(f)
            images.append(np.rint(seed_crop(im,*cc,ss)*255).astype(np.uint8))
            scale=64/(3*ss);affine=np.array([[scale,0,32-scale*cc[0]],[0,scale,32-scale*cc[1]]],np.float32)
            shape=im.shape[:2]
            if shape not in self.ones:self.ones[shape]=np.ones(shape,np.float32)
            support=float(cv2.warpAffine(self.ones[shape],affine,(64,64),flags=cv2.INTER_LINEAR if scale>=1 else cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0).mean())
            logit=float(np.log(np.clip(box[4],1e-6,1-1e-6)/(1-np.clip(box[4],1e-6,1-1e-6)))) if has_link else 0.
            metadata.extend([*( (cc-centre)/side ),np.log(box[2]/current[2]),np.log(box[3]/current[3]),logit,float(has_link),1.,support])
        metadata.extend([np.log(side/20),np.log(width/1000)])
        result=np.asarray(metadata,np.float32);assert result.shape==(26,) and np.isfinite(result).all()
        return np.stack(images),result


class TemporalBirth(nn.Module):
    def __init__(self):
        super().__init__();self.body=SeedVerifier().body
        self.temporal=nn.Sequential(nn.Conv3d(256,128,(3,3,3),padding=(0,1,1)),nn.GroupNorm(8,128),nn.ReLU())
        self.current=nn.Conv2d(256,128,1)
        self.head=nn.Sequential(nn.Linear(538,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,1))
    def freeze_bn_statistics(self):
        for layer in self.body.modules():
            if isinstance(layer,nn.BatchNorm2d):layer.eval()
    def forward(self,clips,metadata,arm='T3'):
        b,t,c,h,w=clips.shape;assert t==3
        valid=metadata[:,:24].reshape(b,3,8)[:,:,6]
        if arm=='S1':clips=clips[:,2:3].expand(-1,3,-1,-1,-1)*valid[:,:,None,None,None]
        elif arm!='T3':raise ValueError(arm)
        z=self.body(clips.reshape(b*3,c,h,w)).reshape(b,3,256,4,4)
        temporal=self.temporal((z*valid[:,:,None,None,None]).transpose(1,2)).squeeze(2)
        joint=torch.cat((self.current(z[:,2]),temporal),1)
        pooled=torch.cat((joint.mean((2,3)),joint.amax((2,3)),metadata),1)
        return self.head(pooled)[:,0]


def augment(clips,metadata,generator):
    """One transform for the entire clip; metadata and missing-image zeros stay consistent."""
    b=len(clips);flip=torch.rand(b,generator=generator)<.5
    clips=clips.clone();metadata=metadata.clone()
    clips[flip]=clips[flip].flip(-1)
    for i in [0,8,16]:metadata[flip,i]*=-1
    gain=.9+.2*torch.rand((b,1,1,1,1),generator=generator)
    return (clips*gain).clamp(0,1),metadata
