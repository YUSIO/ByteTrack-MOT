"""EECTracker PSMP, with explicitly declared EqMotion interface completion.

Eq.(2)'s N,T,f tokens do not match official EqMotion's geometry N,T,2.
We retain the deterministic official geometry/reasoning layers and condition
the invariant node features on the two literal Eq.(2) token sequences. This
bridge is independently chosen; it is not a recovered author implementation.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .vendor.eqmotion.model_t import EqMotion


class ConditionedEqMotion(EqMotion):
    def __init__(self, observed=8, predicted=12, embedding=32, hidden=64, channels=32, layers=4):
        super().__init__(observed, 0, hidden, observed, channels, predicted,
                         device='cpu', n_layers=layers)
        self.position_embedding = nn.Linear(2, embedding, bias=False)
        self.residual_embedding = nn.Linear(2, embedding, bias=False)
        self.swarm_embedding = nn.Linear(2, embedding, bias=False)
        self.condition = nn.Linear(observed * embedding * 2, hidden)

    def forward(self, positions, velocity):
        swarm = velocity.mean(1, keepdim=True)
        c_token = self.position_embedding(positions)
        v_token = self.residual_embedding(velocity-swarm) + self.swarm_embedding(swarm)
        condition = self.condition(torch.cat([c_token, v_token], -1).flatten(2))
        speed = velocity.norm(dim=-1)
        previous = torch.cat([velocity[:, :, :1], velocity[:, :, :-1]], 2)
        cosine = (previous*velocity).sum(-1) / ((previous.norm(dim=-1)+1e-6)*(speed+1e-6))
        angle = torch.acos(cosine.clamp(-1, 1))
        h = torch.cat([self.embedding(speed), self.embedding2(angle)], -1) + condition
        # Below retains the official deterministic N-body EqMotion forward algebra.
        center = positions.mean((1, 2), keepdim=True)
        dct, _ = self.get_dct_matrix(self.in_channel, positions)
        _, idct = self.get_dct_matrix(self.out_channel, positions)
        x = dct @ (positions-center)
        vel = dct @ velocity
        mean = x.mean((-2,-3), keepdim=True)
        x = self.coord_trans((x-mean).transpose(2,3)).transpose(2,3)+mean
        vel = self.vel_trans(vel.transpose(2,3)).transpose(2,3)
        category = self.calc_category(h, torch.cat([x,vel], -2))
        for i in range(self.n_layers):
            h,x,_ = self._modules[f'gcl_{i}'](h,x,vel,category=category)
        mean = x.mean((-2,-3), keepdim=True)
        out = self.predict_head((x-mean).transpose(2,3)).transpose(2,3)+mean
        return idct @ out + center, h


def aggregate(velocity, offset, covariance, beta=1., epsilon=1e-6):
    """Eqs.(5)-(8): covariance between OFFSET means, not full velocities."""
    global_velocity = velocity.mean(1)
    cosine = (velocity * global_velocity[:,None]).sum(-1) / (
        velocity.norm(dim=-1) * global_velocity.norm(dim=-1)[:,None] + epsilon)
    weights = (beta*cosine).softmax(1)
    mean = (weights[...,None]*offset).sum(1)
    delta = offset - mean[:,None]
    cov = (weights[...,None,None] * (covariance + delta[..., :,None]*delta[...,None,:])).sum(1)
    return {'velocity':global_velocity, 'mean':mean, 'covariance':cov, 'weights':weights}


class PSMP(nn.Module):
    def __init__(self, observed=8, predicted=12, embedding=32, hidden=64, channels=32,
                 layers=4, beta=1., coordinate_scale=100., min_std=.01):
        super().__init__()
        self.observed, self.predicted = observed, predicted
        if observed < 2 or predicted < 1 or coordinate_scale <= 0 or min_std <= 0:
            raise ValueError('Invalid PSMP dimensions or units')
        self.scale, self.beta, self.min_std = coordinate_scale, beta, min_std
        self.trajectory = ConditionedEqMotion(observed,predicted,embedding,hidden,channels,layers)
        self.uncertainty = nn.Sequential(nn.Linear(hidden,hidden), nn.SiLU(), nn.Linear(hidden,5))

    def forward(self, positions):
        """[B,N,8,2] observed centers in detector-input pixels; N>0, no padding."""
        if positions.ndim != 4 or positions.shape[2:] != (self.observed,2) or not positions.shape[1]:
            raise ValueError('Expected nonempty observed trajectories [B,N,T,2]')
        if not torch.isfinite(positions).all():
            raise ValueError('Nonfinite trajectory')
        normalized = positions / self.scale
        difference = normalized[:,:,1:] - normalized[:,:,:-1]
        velocity = torch.cat([difference[:,:,:1],difference],2)
        future, hidden = self.trajectory(normalized,velocity)
        future = future * self.scale
        raw = self.uncertainty(hidden)
        offset = raw[...,:2] * self.scale
        factor = raw.new_zeros((*raw.shape[:-1],2,2))
        factor[...,0,0] = F.softplus(raw[...,2]) + self.min_std/self.scale
        factor[...,1,0] = raw[...,3]
        factor[...,1,1] = F.softplus(raw[...,4]) + self.min_std/self.scale
        factor = factor * self.scale
        covariance = factor @ factor.transpose(-1,-2)
        prior = aggregate(future[:,:,0]-positions[:,:,-1],offset,covariance,self.beta)
        return {'future':future, 'offset':offset, 'covariance':covariance,
                'cholesky':factor, 'prior':prior}


def psmp_loss(prediction, future, reference):
    """Eqs.(28)-(31). Reference at t averages GT t-3..t+3, TRAIN ONLY."""
    position = (prediction['future']-future).square().sum(-1).mean()
    target_offset = reference-prediction['future'][:,:,0]
    residual = target_offset-prediction['offset']
    factor = prediction['cholesky']
    whitened = torch.linalg.solve_triangular(factor,residual[...,None],upper=False)[...,0]
    nll = (.5*whitened.square().sum(-1) + factor.diagonal(dim1=-2,dim2=-1).log().sum(-1)
           + torch.log(factor.new_tensor(2*torch.pi))).mean()
    return {'total_loss':position+.01*nll,'position':position,'nll':nll}


def load_psmp(path, config):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state.get('psmp_config') != config:
        raise ValueError('PSMP checkpoint architecture/units do not match config')
    model=PSMP(**config);model.load_state_dict(state['model'],strict=True)
    return model
