import math
import unittest
import torch
from feedback.eec import EECActivation, gaussian_weights, local_entropy, warp
from feedback.psmp import PSMP, aggregate, psmp_loss


class EECTests(unittest.TestCase):
    def test_warp_direction_and_fractional_interpolation(self):
        p=torch.zeros(1,1,5,7);p[0,0,2,2]=1
        actual=warp(p,torch.tensor([[1.,0.]]))
        expected=torch.zeros_like(p);expected[0,0,2,3]=1
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
        half=warp(p,torch.tensor([[.5,0.]]))
        self.assertAlmostEqual(half[0,0,2,2].item(),.5,places=6)
        self.assertAlmostEqual(half[0,0,2,3].item(),.5,places=6)

    def test_entropy_matches_explicit_local_distribution(self):
        energy=torch.randn(1,2,5,6,dtype=torch.float64).square()*20
        actual=local_entropy(energy,3)
        for c in range(2):
            for y in range(5):
                for x in range(6):
                    v=energy[0,c,max(0,y-1):y+2,max(0,x-1):x+2].flatten()
                    logp=(-v).log_softmax(0)
                    expected=-(logp.exp()*logp).sum()
                    self.assertAlmostEqual(actual[0,c,y,x].item(),expected.item(),places=10)
        constant=local_entropy(torch.ones(1,1,4,4),3)
        self.assertAlmostEqual(constant[0,0,1,1].item(),math.log(9),places=6)
        self.assertAlmostEqual(constant[0,0,0,0].item(),math.log(4),places=6)

    def test_expected_squared_residual_not_square_of_expectation(self):
        m=EECActivation(1,window=3,stride=1).double()
        prev=torch.arange(35,dtype=torch.float64).view(1,1,5,7)/10
        vel=torch.zeros(1,2,dtype=torch.float64)
        mean=vel.clone();cov=torch.eye(2,dtype=torch.float64)[None]
        projected,_,_=m.statistics(torch.zeros_like(prev),prev,vel,mean,cov)
        _,energy,_=m.statistics(projected.detach(),prev,vel,mean,cov)
        self.assertGreater(energy[0,0,2,3].item(),.1)
        offsets,weights=gaussian_weights(mean,cov,3)
        samples=torch.stack([warp(prev,vel+d) for d in offsets],0)
        expected=((samples-projected).square()*weights[0,:,None,None,None,None]).sum(0)
        torch.testing.assert_close(energy,expected)

    def test_image_to_feature_covariance_units(self):
        torch.manual_seed(1)
        a=EECActivation(2,window=3,stride=8).double()
        b=EECActivation(2,window=3,stride=1).double()
        x=torch.randn(1,2,5,7,dtype=torch.float64);p=torch.randn_like(x)
        v=torch.tensor([[8.,-4.]],dtype=torch.float64);mu=v/2
        cov=torch.tensor([[[64.,16.],[16.,100.]]],dtype=torch.float64)
        one=a.statistics(x,p,v,mu,cov);two=b.statistics(x,p,v/8,mu/8,cov/64)
        for lhs,rhs in zip(one,two):torch.testing.assert_close(lhs,rhs)

    def test_bypass_and_gradients(self):
        m=EECActivation(2,window=3)
        x=torch.randn(1,2,5,7,requires_grad=True);old=torch.randn_like(x,requires_grad=True)
        prior={'velocity':torch.zeros(1,2),'mean':torch.zeros(1,2),'covariance':torch.eye(2)[None]*64}
        self.assertIs(m(x,None,None),x)
        self.assertTrue(torch.equal(m(x,old,prior),x))
        torch.nn.init.normal_(m.fusion.weight,std=.1)
        out=m(x,old,prior);out.square().mean().backward()
        for p in [x,old,m.alpha,m.tau,m.fusion.weight]:
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(),0)
        with torch.no_grad():m.alpha.fill_(10);m.tau.fill_(-1)
        m.clamp_parameters();self.assertEqual(m.alpha.max().item(),2)
        self.assertAlmostEqual(m.tau.min().item(),.001)


class PSMPTests(unittest.TestCase):
    def test_moment_matching_and_zero_velocity(self):
        v=torch.tensor([[[1.,0.],[3.,0.]]]);offset=torch.tensor([[[-1.,0.],[1.,0.]]])
        cov=torch.eye(2)[None,None].repeat(1,2,1,1)
        p=aggregate(v,offset,cov)
        torch.testing.assert_close(p['velocity'],torch.tensor([[2.,0.]]))
        torch.testing.assert_close(p['mean'],torch.zeros(1,2),atol=1e-6,rtol=0)
        torch.testing.assert_close(p['covariance'],torch.tensor([[[2.,0.],[0.,1.]]]))
        p=aggregate(torch.zeros_like(v),offset,cov)
        torch.testing.assert_close(p['weights'],torch.tensor([[.5,.5]]))

    def test_training_permutation_singleton_and_nll(self):
        torch.manual_seed(6);torch.set_num_threads(2)
        m=PSMP(embedding=8,hidden=16,channels=8,layers=2)
        position=torch.randn(1,3,8,2).cumsum(2)+20
        result=m(position)
        self.assertEqual(result['future'].shape,(1,3,12,2))
        self.assertTrue((torch.linalg.eigvalsh(result['covariance'])>0).all())
        perm=m(position[:,[2,0,1]])
        torch.testing.assert_close(result['future'][:,[2,0,1]],perm['future'],atol=1e-4,rtol=1e-4)
        for key in ('velocity','mean','covariance'):
            torch.testing.assert_close(result['prior'][key],perm['prior'][key],atol=.01,rtol=1e-4)
        singleton=m(torch.ones(1,1,8,2));self.assertTrue(torch.isfinite(singleton['future']).all())
        target=torch.randn(1,3,12,2);ref=torch.randn(1,3,2)
        loss=psmp_loss(result,target,ref)
        normal=torch.distributions.MultivariateNormal(result['offset'],scale_tril=result['cholesky'])
        expected=-normal.log_prob(ref-result['future'][:,:,0]).mean()
        torch.testing.assert_close(loss['nll'],expected)
        loss['total_loss'].backward()
        for name,p in m.named_parameters():
            if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all(),name)
        self.assertGreater(m.trajectory.swarm_embedding.weight.grad.abs().sum().item(),0)
        self.assertGreater(m.uncertainty[-1].weight.grad.abs().sum().item(),0)


if __name__=='__main__':unittest.main()
