"""EECTracker Eqs.(10)-(22), discrete Gaussian expectation and local entropy.

Offsets are sampled around the QUERY pixel (not recentered at the Gaussian
mean). Historical boundary samples use zero padding; entropy excludes padding.
Streaming neighborhoods avoid allocating B*C*H*W*K*K feature tensors.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def gaussian_weights(mean, covariance, window, jitter=1e-6):
    radius=window//2
    axis=torch.arange(-radius,radius+1,device=mean.device,dtype=mean.dtype)
    y,x=torch.meshgrid(axis,axis,indexing='ij')
    delta=torch.stack([x.flatten(),y.flatten()],-1)  # delta = p-u
    residual=delta[None]-mean[:,None]
    covariance=covariance+torch.eye(2,device=mean.device,dtype=mean.dtype)*jitter
    factor=torch.linalg.cholesky(covariance)
    z=torch.linalg.solve_triangular(factor,residual.transpose(1,2),upper=False)
    weights=(-.5*z.square().sum(1)).softmax(1)
    return delta,weights


def warp(previous, displacement):
    b,_,h,w=previous.shape
    y,x=torch.meshgrid(torch.arange(h,device=previous.device,dtype=previous.dtype),
                       torch.arange(w,device=previous.device,dtype=previous.dtype),indexing='ij')
    grid=torch.stack([x,y],-1)[None]-displacement[:,None,None,:]
    grid=(grid+.5)*previous.new_tensor([2/w,2/h])-1
    return F.grid_sample(previous,grid,mode='bilinear',padding_mode='zeros',align_corners=False)


def local_entropy(energy, window):
    """Entropy over neighboring PIXELS of E (Eq.18), not sampling offsets."""
    r=window//2;h,w=energy.shape[-2:]
    minimum=-F.max_pool2d(-energy,window,stride=1,padding=r)
    padded=F.pad(energy,(r,r,r,r),value=float('inf'))
    z=torch.zeros_like(energy); weighted=torch.zeros_like(energy)
    for dy in range(window):
        for dx in range(window):
            difference=padded[:,:,dy:dy+h,dx:dx+w]-minimum
            probability=(-difference).exp()
            finite_difference=torch.where(torch.isfinite(difference),difference,0)
            z=z+probability
            weighted=weighted+probability*finite_difference
    return z.log()+weighted/z


class EECActivation(nn.Module):
    def __init__(self, channels=320, window=7, stride=8, entropy_weight=.1,
                 checkpoint_statistics=True):
        super().__init__()
        if window < 1 or window%2 != 1 or stride <= 0:
            raise ValueError('EEC requires an odd positive neighborhood and positive stride')
        self.window,self.stride,self.entropy_weight=window,stride,entropy_weight
        self.checkpoint_statistics=checkpoint_statistics
        self.alpha=nn.Parameter(torch.ones(1,channels,1,1))
        self.tau=nn.Parameter(torch.full((1,channels,1,1),.8))
        self.fusion=nn.Conv2d(2*channels,channels,3,padding=1)
        # Explicit migration initialization; author initialization is unspecified.
        nn.init.zeros_(self.fusion.weight);nn.init.zeros_(self.fusion.bias)

    @torch.no_grad()
    def clamp_parameters(self):
        self.alpha.clamp_(.001,2.);self.tau.clamp_(.001,2.)

    def statistics(self,current,previous,velocity,mean,covariance):
        offsets,weights=gaussian_weights(mean/self.stride,covariance/self.stride**2,self.window)
        projection=torch.zeros_like(current);energy=torch.zeros_like(current)
        for j,offset in enumerate(offsets):
            sample=warp(previous,velocity/self.stride+offset[None])
            weight=weights[:,j,None,None,None]
            projection=projection+weight*sample
            energy=energy+weight*(sample-current).square()  # E[r^2], never E[r]^2
        entropy=local_entropy(energy,self.window)
        return projection,energy,entropy

    def forward(self,current,previous,prior,return_details=False):
        if previous is None or prior is None:
            return current
        if current.shape != previous.shape or current.shape[1] != self.alpha.shape[1]:
            raise ValueError('Current/previous EEC feature shapes must match')
        args=(current,previous,prior['velocity'].to(current),prior['mean'].to(current),
              prior['covariance'].to(current))
        if self.training and self.checkpoint_statistics and (current.requires_grad or previous.requires_grad):
            projected,energy,entropy=checkpoint(self.statistics,*args,use_reentrant=False)
        else:
            projected,energy,entropy=self.statistics(*args)
        score=energy+self.entropy_weight*entropy
        mask=F.hardsigmoid(self.alpha.clamp(.001,2.)*(self.tau.clamp(.001,2.)-score))
        out=self.fusion(torch.cat([mask*projected,current],1))+current
        if return_details:
            return out,{'projected':projected,'energy':energy,'entropy':entropy,'mask':mask}
        return out
