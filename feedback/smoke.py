"""Real YOLOX-X checkpoint/sequence preflight, no GT, no accuracy claim."""
import argparse
import json
import time
from pathlib import Path
import torch
from .data import read_image, sequence_info
from .model import build, FeedbackDetector
from .online import OnlineFeedback
from .psmp import PSMP
from .infer import sha256


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sequence',default='UAVSwarm-01')
    p.add_argument('--frames',type=int,default=10)
    p.add_argument('--size',type=int,nargs=2,default=[800,1440])
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(160)
    a.output.mkdir(parents=True,exist_ok=False)
    config=json.loads((Path(__file__).parent/'configs/eec_v1.json').read_text())
    config['input_size']=a.size
    baseconfig={**config,'mode':'none'}
    base=build(baseconfig,a.checkpoint).eval()
    prompt=FeedbackDetector(base.detector,'pp').eval()
    eec=FeedbackDetector(base.detector,'eec',eec_config=config['eec']).eval()
    predictor=PSMP(**config['psmp']).eval()
    path=a.data_root/'train'/a.sequence;info=sequence_info(path)
    report={'kind':'preflight','gt_read':False,'feedback_weights':'random_identity_initialized',
            'psmp_weights':'random','input_size':a.size,'architecture':'YOLOX-X',
            'checkpoint_sha256':sha256(a.checkpoint),'frames':a.frames,'arms':{}}
    baseline_rows=None
    for mode,model in [('none',base),('pp',prompt),('eec',eec)]:
        run=OnlineFeedback(model,{**config,'mode':mode},predictor if mode=='eec' else None,int(info['framerate']))
        rows=[];trace=[];start=time.perf_counter()
        for frame in range(1,a.frames+1):
            image,ratio,shape=read_image(path,frame,info,a.size)
            result=run.step(image[None],shape,ratio)
            rows.append({'tracks':result['tracks'],'detections':result['detections'].tolist()})
            trace.append({k:v for k,v in result.items() if k not in ('tracks','detections')})
        if baseline_rows is None:baseline_rows=rows
        else:assert rows==baseline_rows,(mode,'zero-initialized feedback must preserve outputs')
        report['arms'][mode]={'seconds':time.perf_counter()-start,
                             'feedback_frames':sum(r['feedback_used'] for r in trace),
                             'same_as_baseline':rows==baseline_rows,
                             'detections':sum(len(r['detections']) for r in rows),
                             'track_rows':sum(len(r['tracks']) for r in rows),'trace':trace}
        print(json.dumps({'mode':mode,**{k:v for k,v in report['arms'][mode].items() if k!='trace'}}),flush=True)
    # Nonzero residuals must reach actual X outputs, without using GT prompts.
    image,ratio,shape=read_image(path,1,info,a.size)
    with torch.no_grad():
        original=base(image[None])
        boxes=baseline_rows[0]['detections']
        if not boxes:raise RuntimeError('Select a preflight frame with a real detection')
        b=torch.tensor(boxes[:1])[:,:4];b[:,2:]-=b[:,:2];b*=ratio
        torch.nn.init.normal_(prompt.feedback.to_patch.weight,std=1e-4)
        changed=prompt(image[None],contexts=[{'boxes':b,'ids':torch.tensor([1])}])
        pp_diff=(changed-original).abs().max().item()
        features,_,_=base.features(image[None])
        old=features[0].clone()
        torch.nn.init.normal_(eec.feedback.fusion.weight,std=1e-4)
        prior={'velocity':torch.zeros(1,2),'mean':torch.zeros(1,2),'covariance':torch.eye(2)[None]*64}
        changed=eec(image[None],contexts=[{'prior':prior,'previous':old}])
        eec_diff=(changed-original).abs().max().item()
        assert pp_diff>0 and eec_diff>0
        assert torch.isfinite(changed).all()
    report['nonzero_residual_max_raw_delta']={'pp':pp_diff,'eec':eec_diff}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'nonzero_residual_max_raw_delta':report['nonzero_residual_max_raw_delta'],'status':'passed'}))


if __name__=='__main__':main()
