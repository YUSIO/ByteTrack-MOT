"""Causal train/dev inference and fixed COCO detection evaluation, never test."""
import contextlib
import io
import json
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from feedback.data import read_image, sequence_info
from .online import TraDeSOnline

NAMES=['AP50:95','AP50','AP75','AP_small','AP_medium','AP_large',
       'AR1','AR10','AR100','AR_small','AR_medium','AR_large']


def evaluate(cache, root, sequences):
    images, annotations, detections=[],[],[]
    for name in sequences:
        path=root/'train'/name;info=sequence_info(path)
        width,height,length=[int(info[k]) for k in ('imwidth','imheight','seqlength')]
        base=len(images)
        images.extend(dict(id=base+i,width=width,height=height) for i in range(1,length+1))
        for row in np.loadtxt(path/'gt/gt.txt',delimiter=',',ndmin=2):
            if row[6]<1:continue
            frame,_,x,y,w,h=row[:6]
            x1,y1,x2,y2=max(x,0),max(y,0),min(x+w,width),min(y+h,height)
            if x2>x1 and y2>y1:
                annotations.append(dict(id=len(annotations)+1,image_id=base+int(frame),
                    category_id=1,iscrowd=0,bbox=[x1,y1,x2-x1,y2-y1],area=(x2-x1)*(y2-y1)))
        for line in (cache/name/'det.txt').read_text().splitlines():
            v=list(map(float,line.split(',')))
            if int(v[0])<1 or int(v[0])>length:raise ValueError('Detection frame outside sequence')
            detections.append(dict(image_id=base+int(v[0]),category_id=1,bbox=v[2:6],score=v[6]))
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO();gt.dataset=dict(images=images,annotations=annotations,categories=[dict(id=1,name='UAV')]);gt.createIndex()
        if detections:dt=gt.loadRes(detections)
        else:
            dt=COCO();dt.dataset=dict(images=images,annotations=[],categories=gt.dataset['categories']);dt.createIndex()
        ev=COCOeval(gt,dt,'bbox');ev.evaluate();ev.accumulate();ev.summarize()
    # COCO's own score-ordered one-to-one matches, fixed IoU .5 / score >= .6.
    tp=fp=0
    for item in ev.evalImgs:
        if item is None or item['aRng']!=ev.params.areaRng[0]:continue
        selected=(np.asarray(item['dtScores'])>=.6)&~item['dtIgnore'][0]
        tp+=int(((item['dtMatches'][0]>0)&selected).sum())
        fp+=int(((item['dtMatches'][0]==0)&selected).sum())
    return dict(metrics=dict(zip(NAMES,map(float,ev.stats))),frames=len(images),
        gt_boxes=len(annotations),detections=len(detections),sequences=sequences,
        fixed_iou50_score06=dict(TP=tp,FP=fp,FN=len(annotations)-tp,recall=tp/max(len(annotations),1)),
        evaluator='pycocotools COCOeval bbox; maxDets=1,10,100; all sequence frames; clipped GT conf>=1')


@torch.no_grad()
def validate(model, root, sequences, output, mode, device):
    if mode not in ('on','off','heat_shift'):raise ValueError(mode)
    output.mkdir(parents=True,exist_ok=False)
    traces=[]
    for name in sequences:
        runner=TraDeSOnline(model,enabled=mode!='off',heat_shift=mode=='heat_shift');runner.reset(name)
        path=root/'train'/name;info=sequence_info(path);out=output/name;out.mkdir()
        used=0
        with (out/'det.txt').open('w') as det,(out/'tracks.txt').open('w') as tracks:
            for frame in range(1,int(info['seqlength'])+1):
                image,ratio,shape=read_image(path,frame,info,model.input_size)
                r=runner.step(image[None].to(device),name,frame,shape,ratio)
                used+=int(r['feedback_used'])
                if not np.isfinite(r['detections']).all():raise FloatingPointError('Nonfinite detections')
                for x1,y1,x2,y2,score in r['detections']:
                    det.write(f'{frame},-1,{x1:.4f},{y1:.4f},{x2-x1:.4f},{y2-y1:.4f},{score:.6f},-1,-1,-1\n')
                for identity,x,y,w,h,score in r['tracks']:
                    tracks.write(f'{frame},{identity},{x:.1f},{y:.1f},{w:.1f},{h:.1f},{score:.6f},-1,-1,-1\n')
        result=evaluate(output,root,[name]);result['feedback_frames']=used
        (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');traces.append(result)
        print(json.dumps(dict(phase='validation',mode=mode,sequence=name,metrics=result['metrics'],feedback_frames=used)),flush=True)
    result=evaluate(output,root,sequences);result.update(mode=mode,per_sequence=traces)
    (output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
