"""Tracking from frozen detections and pixels only; deliberately no GT reads."""
import argparse
import configparser
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from .model import MPANet
from .data import crop_tensor, image_path
from .association import HOMAAssociation
from .tracker import MHATracker
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.basetrack import BaseTrack


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True); p.add_argument('--detections',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True); p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--split',choices=['train','test'],required=True)
    p.add_argument('--arm',choices=['byte','mha_iou','mpa_iou','m2da','homa'],required=True)
    p.add_argument('--window',type=int,default=8); p.add_argument('--motion-window',type=int,default=8)
    p.add_argument('--kappa',type=float,default=.1); p.add_argument('--match-thresh',type=float,default=.9)
    p.add_argument('--reduction',choices=['sum','mean'],default='sum'); p.add_argument('--sequences',nargs='+')
    p.add_argument('--feature-cache',type=Path)
    args=p.parse_args(); cfg=json.loads(args.config.read_text()); torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=False)
    names=args.sequences or sorted(x.name for x in args.detections.glob('UAVSwarm-*') if x.is_dir())
    appearance=args.arm in ('mpa_iou','homa'); model=None; cksha=None
    if appearance:
        assert args.checkpoint and torch.cuda.is_available()
        cksha=hashlib.file_digest(args.checkpoint.open('rb'),'sha256').hexdigest()
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        architecture=state['config']['model'].get('architecture','legacy_v1')
        assert architecture==cfg['model'].get('architecture','legacy_v1'), 'checkpoint architecture/config mismatch'
        model=MPANet(architecture=architecture).cuda().eval(); model.load_state_dict(state['model'])
    summary=[]
    for name in names:
        started=time.time(); seq=args.data/args.split/name
        ini=configparser.ConfigParser(); ini.read(seq/'seqinfo.ini'); info=ini['Sequence']
        length,height,width,fps=(int(info[k]) for k in ('seqLength','imHeight','imWidth','frameRate'))
        detpath=args.detections/name/'det.txt'; raw=np.loadtxt(detpath,delimiter=',',ndmin=2)
        assert np.isfinite(raw).all() and (raw[:,0]>=1).all() and (raw[:,0]<=length).all()
        detsha=hashlib.file_digest(detpath.open('rb'),'sha256').hexdigest()
        tracker_class=MHATracker if args.arm!='byte' and cfg.get('lifecycle')=='mha_v2' else BYTETracker
        tracker=tracker_class(SimpleNamespace(**(cfg['tracker']|{'match_thresh':args.match_thresh})),frame_rate=fps); BaseTrack._count=0
        if args.arm!='byte': tracker.homa=HOMAAssociation(model,args.arm,args.window,args.motion_window,args.kappa,args.reduction,cfg['tracker']['track_thresh'])
        cached={}; cachepath=None
        if appearance and args.feature_cache:
            args.feature_cache.mkdir(parents=True,exist_ok=True)
            cachepath=args.feature_cache/f'{name}-{cksha[:16]}-{detsha[:16]}.pt'
            if cachepath.exists():
                obj=torch.load(cachepath,map_location='cpu',weights_only=False)
                assert obj['checkpoint_sha256']==cksha and obj['detector_sha256']==detsha
                cached=obj['frames']
        output=[]; computed=False
        for frame in range(1,length+1):
            r=raw[raw[:,0]==frame]
            det=np.column_stack((r[:,2:4],r[:,2:4]+r[:,4:6],r[:,6])).astype(np.float32)
            if appearance:
                high=r[det[:,4]>cfg['tracker']['track_thresh']]
                if frame in cached:
                    features=cached[frame].cuda()
                elif len(high):
                    with Image.open(image_path(seq,frame)) as im: crops=crop_tensor(im.convert('RGB'),high[:,2:6]).cuda()
                    with torch.inference_mode(),torch.autocast('cuda'): features=model.encode(crops)
                    cached[frame]=features.detach().cpu(); computed=True
                else:features=torch.empty((0,8192),device='cuda')
                tracker.homa.set_current(features,frame)
            elif args.arm in ('m2da','mha_iou'):tracker.homa.set_current(None,frame)
            tracks=tracker.update(det.copy(),(height,width),(height,width))
            for t in tracks:
                x,y,w,h=t.tlwh
                if w*h>cfg['tracker']['min_box_area'] and w/h<=cfg['tracker']['aspect_ratio_thresh']:
                    output.append(f'{frame},{t.track_id},{round(x,1)},{round(y,1)},{round(w,1)},{round(h,1)},{round(t.score,2)},-1,-1,-1\n')
        (args.output/f'{name}.txt').write_text(''.join(output))
        if cachepath and computed:
            torch.save({'checkpoint_sha256':cksha,'detector_sha256':detsha,'frames':cached},cachepath)
        summary.append({'sequence':name,'frames':length,'detections':len(raw),'rows':len(output),'seconds':time.time()-started,'detector_sha256':detsha,'track_sha256':hashlib.file_digest((args.output/f'{name}.txt').open('rb'),'sha256').hexdigest()})
        if isinstance(tracker,MHATracker):
            summary[-1]['lifecycle_counts']=dict(tracker.audit_counts)
            summary[-1]['excluded_low_events']=tracker.audit_events
        print(json.dumps(summary[-1]),flush=True)
    (args.output/'summary.json').write_text(json.dumps({'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'checkpoint_sha256':cksha,'gt_read':False,'sequences':summary},indent=2)+'\n')


if __name__=='__main__': main()
