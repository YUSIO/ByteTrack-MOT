"""Train-split closed-loop validation: predictions are generated before GT is read."""
import contextlib
import io
import json
from pathlib import Path
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from .data import read_image, sequence_info
from .online import OnlineFeedback
from .infer import sha256

NAMES = ['AP50:95','AP50','AP75','AP_small','AP_medium','AP_large',
         'AR1','AR10','AR100','AR_small','AR_medium','AR_large']


def evaluate(cache, root, sequences):
    images, annotations, detections, hashes = [], [], [], {}
    for name in sequences:
        path = root/name
        info = sequence_info(path)
        width,height,length = [int(info[k]) for k in ('imwidth','imheight','seqlength')]
        base = len(images)
        images.extend({'id':base+i,'width':width,'height':height} for i in range(1,length+1))
        gt_path = path/'gt/gt.txt'
        hashes[name] = sha256(gt_path)
        for row in np.loadtxt(gt_path,delimiter=',',ndmin=2):
            if row[6] < 1:
                continue
            frame,_,x,y,w,h = row[:6]
            x1,y1,x2,y2 = max(x,0),max(y,0),min(x+w,width),min(y+h,height)
            if x2>x1 and y2>y1:
                annotations.append({'id':len(annotations)+1,'image_id':base+int(frame),
                    'category_id':1,'iscrowd':0,'bbox':[x1,y1,x2-x1,y2-y1], 'area':(x2-x1)*(y2-y1)})
        for line in (cache/name/'det.txt').read_text().splitlines():
            v = list(map(float,line.split(',')))
            detections.append({'image_id':base+int(v[0]),'category_id':1,'bbox':v[2:6],'score':v[6]})
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO();gt.dataset={'images':images,'annotations':annotations,'categories':[{'id':1,'name':'UAV'}]};gt.createIndex()
        if detections:
            dt=gt.loadRes(detections)
        else:
            dt=COCO();dt.dataset={'images':images,'annotations':[],'categories':gt.dataset['categories']};dt.createIndex()
        ev=COCOeval(gt,dt,'bbox');ev.evaluate();ev.accumulate();ev.summarize()
    result={'metrics':dict(zip(NAMES,map(float,ev.stats))),'frames':len(images),'gt_boxes':len(annotations),
            'detections':len(detections),'sequences':list(sequences),'gt_sha256':hashes,
            'evaluator':'pycocotools COCOeval bbox maxDets=1,10,100; GT clipped; conf>=1'}
    (cache/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


@torch.no_grad()
def validate(model, config, predictor, root, sequences, output, device):
    output.mkdir(parents=True,exist_ok=False)
    for name in sequences:
        path=root/'train'/name;info=sequence_info(path)
        runner=OnlineFeedback(model,config,predictor,int(info['framerate']))
        out=output/name;out.mkdir()
        with (out/'det.txt').open('w') as det, (out/'tracks.txt').open('w') as tracks:
            for frame in range(1,int(info['seqlength'])+1):
                image,ratio,shape=read_image(path,frame,info,config['input_size'])
                result=runner.step(image[None].to(device),shape,ratio)
                for x1,y1,x2,y2,score in result['detections']:
                    det.write(f'{frame},-1,{x1:.4f},{y1:.4f},{x2-x1:.4f},{y2-y1:.4f},{score:.6f},-1,-1,-1\n')
                for identity,x,y,w,h,score in result['tracks']:
                    tracks.write(f'{frame},{identity},{x:.1f},{y:.1f},{w:.1f},{h:.1f},{score:.6f},-1,-1,-1\n')
    return evaluate(output,root/'train',sequences)
