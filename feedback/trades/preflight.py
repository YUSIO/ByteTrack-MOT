"""Bounded real YOLOX-X implementation checks; diagnostic weights never seed training."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
import torch
import torchvision
from feedback.model import FeedbackDetector
from feedback.online import OnlineFeedback
from feedback.data import read_image
from .model import FrozenTraDeS
from .data import TrainPairs,cva_loss
from .online import TraDeSOnline
from .checkpoint import sha256,save_adapter,load_adapter


def tensor_hashes(module):
    return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for k,v in module.state_dict().items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(2)
    base_hash=sha256(args.checkpoint)
    if base_hash!='e6a7c579fa78e19d331312a2a2324ca594f666112c9f20219f30621db4c3fc21':
        raise ValueError('Expected immutable Exp141 epoch80 detector')
    dirty=subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    if dirty:
        raise RuntimeError('Commit implementation before real preflight')
    sequence='UAVSwarm-05'
    paths=[args.checkpoint,args.data_root/'train'/sequence/'seqinfo.ini',
           args.data_root/'train'/sequence/'gt/gt.txt']
    paths.extend(args.data_root/'train'/sequence/'img1'/f'{f:06d}.jpg' for f in [1,2,3])
    input_hashes={str(p):sha256(p) for p in paths}
    manifest=dict(phase='bounded_real_implementation_preflight',formal_training=False,
        test_read=False,device='cpu',threads=2,seed=0,sequence=sequence,frames=[1,2,3],
        optimizer_steps=2,lr=.0001,momentum=.9,weight_decay=.0001,
        python=sys.version,python_executable=sys.executable,torch=str(torch.__version__),
        torchvision=str(torchvision.__version__),platform=platform.platform(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code_dirty=False,input_hashes=input_hashes,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    model=FrozenTraDeS(args.checkpoint)
    frozen=tensor_hashes(model.detector)
    data=TrainPairs(args.data_root,[sequence])
    samples=[data[i] for i in [0,1]]
    model.eval()
    sample=samples[0];x=sample['image'][None];old=sample['previous'][None];heat=sample['heat'][None]
    with torch.no_grad():
        original=model.detector(x)
        initial=model(x,previous_images=old,heat=heat)['output']
    assert torch.equal(original,initial),'Nonidentity initialization'
    optimizer=torch.optim.SGD(model.optimizer_parameters(),lr=.0001,momentum=.9,weight_decay=.0001)
    train_ids={id(p) for g in optimizer.param_groups for p in g['params']}
    assert train_ids=={id(p) for p in model.adapter.parameters()}
    history=[]
    for step,sample in enumerate(samples,1):
        model.train();optimizer.zero_grad(set_to_none=True)
        result=model(sample['image'][None],previous_images=sample['previous'][None],
                     heat=sample['heat'][None],targets=sample['targets'][None])
        auxiliary=cva_loss(result['cva'],[sample['cva']])
        detection=result['output']['total_loss']
        loss=detection+.5*auxiliary
        assert torch.isfinite(loss)
        loss.backward()
        gradients={n:float(p.grad.abs().sum()) for n,p in model.adapter.named_parameters() if p.grad is not None}
        assert gradients['output.weight']>0
        for name,p in model.adapter.named_parameters():
            assert p.grad is None or torch.isfinite(p.grad).all(),name
        assert all(p.grad is None for p in model.detector.parameters())
        optimizer.step()
        assert tensor_hashes(model.detector)==frozen,'Detector or BN changed'
        history.append(dict(step=step,frame=sample['frame'],detection_loss=float(detection.detach()),
            cva_loss=float(auxiliary.detach()),total_loss=float(loss.detach()),
            pairs=sample['pair_stats'],gradient_l1=gradients))
        print(json.dumps({k:v for k,v in history[-1].items() if k!='gradient_l1'}),flush=True)
        del result,auxiliary,detection,loss
    model.eval()
    with torch.no_grad():
        off=model(x,previous_images=old,heat=heat,enabled=False)['output']
        empty=model(x,previous_images=old,heat=heat*0)['output']
        on=model(x,previous_images=old,heat=heat)['output']
    assert torch.equal(original,off) and torch.equal(original,empty)
    assert torch.isfinite(on).all()
    raw_delta=float((on-original).abs().max())
    assert raw_delta>0,'Feedback did not change any raw prediction'
    checkpoint=args.output/'diagnostic_adapter.pt'
    save_adapter(model,checkpoint,args.checkpoint,dict(diagnostic_only=True,formal_initialization_forbidden=True))
    saved=tensor_hashes(model.adapter)
    with torch.no_grad():model.adapter.output.weight.zero_()
    load_adapter(model,checkpoint,args.checkpoint)
    assert saved==tensor_hashes(model.adapter)
    with torch.no_grad():reloaded=model(x,previous_images=old,heat=heat)['output']
    assert torch.equal(on,reloaded)
    # Closed-loop OFF parity against the inherited canonical ByteTrack wrapper.
    config={'tracker':dict(track_thresh=.6,track_buffer=30,match_thresh=.9,mot20=False),
            'conf':.001,'nms':.7}
    baseline=OnlineFeedback(FeedbackDetector(model.detector,'none'),config)
    disabled=TraDeSOnline(model,enabled=False);disabled.reset(sequence)
    path,info,_=data.data[sequence];traces=[]
    for frame in [1,2,3]:
        image,ratio,shape=read_image(path,frame,info,(800,1440))
        a=baseline.step(image[None],shape,ratio)
        b=disabled.step(image[None],sequence,frame,shape,ratio)
        assert np.array_equal(a['detections'],b['detections'])
        assert a['tracks']==b['tracks']
        traces.append(dict(frame=frame,detections=len(b['detections']),tracks=len(b['tracks'])))
    enabled=TraDeSOnline(model);enabled.reset(sequence);online=[]
    for frame in [1,2,3]:
        image,ratio,shape=read_image(path,frame,info,(800,1440))
        r=enabled.step(image[None],sequence,frame,shape,ratio)
        online.append(dict(frame=frame,feedback_used=r['feedback_used'],history_frame=r['history_frame']))
    assert not online[0]['feedback_used'] and any(r['feedback_used'] for r in online[1:])
    assert tensor_hashes(model.detector)==frozen
    assert input_hashes=={str(p):sha256(p) for p in paths}
    metrics=dict(status='passed',initial_raw_identity=True,detector_tensors_unchanged=len(frozen),
        feedback_parameters=sum(p.numel() for p in model.adapter.parameters()),
        optimizer_adapter_only=True,updates=history,off_and_empty_raw_bitwise_identical=True,
        on_max_raw_delta=raw_delta,adapter_reload_bitwise_identical=True,
        off_detector_and_track_parity=traces,causal_online=online,
        diagnostic_checkpoint_sha256=sha256(checkpoint),performance_claim=False)
    (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    manifest.update(status='passed',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in metrics.items() if k!='updates'}),flush=True)


if __name__=='__main__':main()
