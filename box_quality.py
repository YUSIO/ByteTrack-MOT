"""Single-frame candidate IoU regression; shared local image and ROI head, no history."""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def pack_pair(image, original, candidate):
    boxes=np.stack([original,candidate]).astype(np.float32)
    lo=boxes[:,:2].min(0);hi=(boxes[:,:2]+boxes[:,2:]).max(0)
    center=(lo+hi)/2;side=max(16.,float((hi-lo).max())*2)
    scale=96/side;origin=center-side/2
    matrix=np.array([[scale,0,-scale*origin[0]],[0,scale,-scale*origin[1]]],np.float32)
    patch=cv2.warpAffine(image,matrix,(96,96),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    xy=(boxes[:,:2]-origin)/side;wh=boxes[:,2:]/side
    return np.ascontiguousarray(patch[:,:,::-1].transpose(2,0,1)),np.concatenate([xy,xy+wh],axis=1).astype(np.float32)

class QualityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(3,32,3,1,1),nn.GroupNorm(8,32),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.GroupNorm(8,64),nn.ReLU(),nn.Conv2d(64,96,3,1,1),nn.GroupNorm(8,96),nn.ReLU(),nn.Conv2d(96,96,3,1,1),nn.GroupNorm(8,96),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(96*5*5+4,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self,images,boxes):
        feature=self.body(images)
        n,k,_=boxes.shape
        a=(torch.arange(5,device=boxes.device,dtype=boxes.dtype)+.5)/5
        xx=boxes[:,:,0,None]+(boxes[:,:,2]-boxes[:,:,0])[:,:,None]*a
        yy=boxes[:,:,1,None]+(boxes[:,:,3]-boxes[:,:,1])[:,:,None]*a
        grid=torch.stack([xx[:,: ,None,:].expand(n,k,5,5),yy[:,:,:,None].expand(n,k,5,5)],-1)*2-1
        # Grid centers use the common crop coordinate system, preserving box differences.
        roi=F.grid_sample(feature,grid.reshape(n,k*5,5,2),align_corners=False,padding_mode='border')
        roi=roi.reshape(n,96,k,5,5).permute(0,2,1,3,4).reshape(n,k,-1)
        return torch.sigmoid(self.head(torch.cat([roi,boxes],-1)).squeeze(-1))

@torch.no_grad()
def quality_scores(model,packed):
    image,boxes=packed;device=next(model.parameters()).device
    return model(torch.from_numpy(image[None]).to(device).float()/255,torch.from_numpy(boxes[None]).to(device))[0].cpu().numpy()
