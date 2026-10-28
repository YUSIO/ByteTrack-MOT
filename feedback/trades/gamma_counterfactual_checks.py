"""Bounded algebra and sampler checks; synthetic inputs, no dataset access."""
from types import SimpleNamespace
import unittest
import torch
from .gamma_counterfactual import split_offsets
from .vendor.base_model import BaseModel
from .vendor.ops import DCN_TraDeS


class Checks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(2)
        opt=SimpleNamespace(head_kernel=3,trades=True,output_h=8,output_w=10,
                            deform_kernel_size=3,device='cpu',clip_len=2)
        self.core=BaseModel({}, {}, 1, 64, opt).eval()
        self.current=torch.randn(1,64,8,10)
        self.previous=torch.randn_like(self.current)
        self.d=torch.randn(1,2,8,10)

    def test_exact_affine_remainder(self):
        c=self.core
        native=c.gamma(self.d,self.current,self.previous,1,8,10)
        modes,r=split_offsets(c,self.d,native)
        zero=c.gamma(torch.zeros_like(self.d),self.current,self.previous,1,8,10)
        torch.testing.assert_close(r,zero,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(modes['C_identity_gain']-modes['A_native'],
                                   modes['D_center_anchor']-modes['B_no_neighbors'],atol=1e-6,rtol=1e-5)

    def test_neighbor_isolation_keeps_dc_gain(self):
        c=self.core
        native=c.gamma(self.d,self.current,self.previous,1,8,10)
        modes,_=split_offsets(c,self.d,native)
        altered=self.d.clone()
        altered[:,:,2:5,3:6]=self.d[:,:,3:4,4:5]
        explicit=c.gamma(altered,self.current,self.previous,1,8,10)
        torch.testing.assert_close(modes['B_no_neighbors'][:,:,3,4],explicit[:,:,3,4],atol=1e-6,rtol=1e-5)

    def test_neighbor_change_does_not_change_isolated_modes(self):
        c=self.core
        changed=self.d.clone();changed[:,:,2,3]+=200
        a,_=split_offsets(c,self.d,c.gamma(self.d,self.current,self.previous,1,8,10))
        b,_=split_offsets(c,changed,c.gamma(changed,self.current,self.previous,1,8,10))
        for mode in ('B_no_neighbors','D_center_anchor'):
            torch.testing.assert_close(a[mode][:,:,3,4],b[mode][:,:,3,4],atol=1e-5,rtol=1e-4)
        self.assertGreater(float((a['A_native'][:,:,3,4]-b['A_native'][:,:,3,4]).abs().max()),1)

    def test_dcn_xy_direction_and_zero_support(self):
        op=DCN_TraDeS(1,1,3,1,1)
        with torch.no_grad():
            op.weight.zero_();op.weight[0,0,1,1]=1;op.bias.fill_(.5)
        field=torch.arange(80,dtype=torch.float32).reshape(1,1,8,10)
        offset=torch.zeros(1,18,8,10);offset[:,8]=1;offset[:,9]=2
        mask=torch.ones(1,9,8,10)
        actual=op(field,offset,mask)
        self.assertAlmostEqual(float(actual[0,0,3,4]),float(field[0,0,4,6])+.5)
        torch.testing.assert_close(op(field*0,offset,mask)-op.bias[None,:,None,None],field*0,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
