"""Synthetic gate checks plus optional two-pair, GT-free GPU software smoke."""
import argparse,json,os,subprocess,unittest
from pathlib import Path
import numpy as np
import torch
from .dense_gate import reliability,recipient_mask,geometry,residuals,blend


class Checks(unittest.TestCase):
    def test_reciprocal_and_unsupported_endpoint(self):
        n=9;cost=torch.eye(n)[None]*10;d=torch.zeros(1,2,6,6)
        r,_=reliability(cost,d,(48,48))
        self.assertGreater(float(r.min()),.999)
        # Native endpoint contradicts the joint peak: endpoint support must reject.
        bad=d.clone();bad[:,0]=2
        r,_=reliability(cost,bad,(48,48));self.assertLess(float(r.max()),1e-10)

    def test_mutually_wrong_correspondence_can_pass(self):
        n=9;p=torch.arange(n);p[0],p[8]=8,0
        cost=torch.full((1,n,n),-10.);cost[0,torch.arange(n),p]=10
        q=torch.arange(n);qx=q%3;qy=q//3;px=p%3;py=p//3
        d=torch.stack(((px-qx)*2,(py-qy)*2)).float().reshape(1,2,3,3)
        d=torch.nn.functional.interpolate(d,scale_factor=2,mode='nearest')
        r,_=reliability(cost,d,(48,48));self.assertGreater(float(r.min()),.999)

    def test_padding_and_soft_score(self):
        det=np.array([[0,0,8,8,.53],[8,0,16,8,.9]])
        m=recipient_mask(det,1,(4,4),(8,16),'cpu')
        self.assertAlmostEqual(float(m[0,0,0,0]),.53,places=6)
        self.assertAlmostEqual(float(m[0,0,0,1]),.9,places=6)
        self.assertEqual(float(m[:,:,1:].sum()),0)
        cost=torch.eye(4)[None]*10;r,_=reliability(cost,torch.zeros(1,2,4,4),(16,32))
        self.assertEqual(float(r[:,:,2:].sum()),0)

    def test_blend_endpoints(self):
        raw,a,d=torch.randn(3,2,4,4),torch.randn(3,2,4,4),torch.randn(3,2,4,4)
        torch.testing.assert_close(blend(raw,a,d,0),raw+a)
        torch.testing.assert_close(blend(raw,a,d,1),raw+d)


@torch.no_grad()
def smoke(root,out):
    from feedback.data import sequence_info,read_image
    from .model import FrozenTraDeS
    from .checkpoint import load_adapter,sha256
    from .data import render_heat
    from .preflight import tensor_hashes
    from .history_counterfactual import detection_lines
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='GPU-743a7bc5-49b1-091e-efe7-fd88ae1b9b9f'
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    base=root/'experiments/exp141_bytetrack_yolox_x_uavswarmv1/results/run_001/selected.pt'
    adapter=root/'experiments/exp194_visual_history_residual/results/run_001/last.pt'
    assert sha256(adapter)=='6ee5a7706dc0bac0e90e9652166d954142681215a5438dbcbca021ea9cd63c8a'
    model=FrozenTraDeS(base).cuda().eval().requires_grad_(False);load_adapter(model,adapter,base)
    model.detector.head.decode_in_inference=False
    before=tensor_hashes(model);rows=[]
    for name,frame in [('UAVSwarm-21',46),('UAVSwarm-63',60)]:
        path=root/'datasets/UAVSwarm-dataset-master/train'/name;info=sequence_info(path)
        image,ratio,shape=read_image(path,frame,info,model.input_size)
        old,_,_=read_image(path,frame-1,info,model.input_size)
        f=model.detector.backbone(image[None].cuda());prev=model.detector.backbone(old[None].cuda())[0]
        valid=(int(shape[0]*ratio),int(shape[1]*ratio))
        geo=geometry(model.adapter,f[0],prev,valid)
        # Artificial box from image geometry, no GT or test input.
        boxes=np.array([[shape[1]*.25,shape[0]*.25,shape[1]*.75,shape[0]*.75]],np.float32)
        heat=render_heat(boxes*ratio,model.input_size)[None].cuda()
        a,d,attention=residuals(model.adapter,geo,heat)
        native=model.adapter.core.CVA_MFW([geo['cur'],geo['prev']],[geo['prev']*heat],visual_only=True)
        e=float((a-model.adapter.output(native[0][0])).abs().max())
        de=float((geo['displacement']-native[2]).abs().max())
        assert e<2e-5 and de<5e-5,(e,de)
        detline=detection_lines(model.detector.head(f),model.detector.head,ratio,frame)
        det=np.array([[*map(float,line.split(',')[2:6]),float(line.split(',')[6])] for line in detline.splitlines()],np.float64).reshape(-1,5)
        det[:,2:4]+=det[:,:2]
        m=recipient_mask(det,ratio,f[0].shape[-2:],valid,f[0].device)
        g=m*geo['reliability'];assert bool(torch.isfinite(g).all()) and float(g.min())>=0 and float(g.max())<=1
        prediction=model.detector.head((blend(f[0],a,d,g),*f[1:]));assert bool(torch.isfinite(prediction).all())
        za,zd,_=residuals(model.adapter,geo,heat*0)
        assert float(za.abs().max())==0 and float(zd.abs().max())==0
        rows.append(dict(sequence=name,frame=frame,native_residual_error=e,motion_error=de,gate_min=float(g.min()),gate_max=float(g.max()),**geo['stats']))
    assert before==tensor_hashes(model)
    manifest=dict(status='completed',phase='software_preflight',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),optimizer_steps=0,test_access=False,gt_access=False,weights_BN_unchanged=True,rows=rows)
    (out/'manifest.yaml').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'exit_code.txt').write_text('0\n')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--smoke-output',type=Path);args=p.parse_args()
    if args.root:smoke(args.root,args.smoke_output)
    else:unittest.main(argv=['dense_gate_checks'])
