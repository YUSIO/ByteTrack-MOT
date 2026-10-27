"""Epoch-ten paired dev history interventions; inference only, fixed cache replay."""
import argparse
import contextlib
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import torch
import torchvision
from feedback.data import read_train_sequences, read_image
from feedback.online import decode
from feedback.state import observe_detector_boxes
from .checkpoint import load_adapter, sha256
from .data import render_heat
from .model import FrozenTraDeS
from .online import observed_history_xyxy
from .preflight import tensor_hashes
from .train import verify_inputs
from .validate import evaluate

MODES = ('off', 'gt', 'online', 'zero_support')


@contextlib.contextmanager
def zero_history_support(core, enabled):
    """Keep CVA, offsets, DCN bias and attention; intervene only on DCN support."""
    def zero(module, args):
        support, offset, mask = args
        return support * 0, offset, mask
    handle = core.dcn1_1.register_forward_pre_hook(zero) if enabled else None
    try:
        yield
    finally:
        if handle is not None: handle.remove()


def cache_detections(path, length):
    frames = {i: [] for i in range(1, length+1)}
    for line in path.read_text().splitlines():
        row = list(map(float, line.split(','))); frame = int(row[0])
        if frame not in frames: raise ValueError('Cache frame outside sequence')
        x, y, w, h, score = row[2:7]
        frames[frame].append([x, y, x+w, y+h, score])
    return {i: np.asarray(v, np.float64).reshape(-1, 5) for i, v in frames.items()}


class HistoryReplay:
    """Replay frozen ON detections, never feed intervention predictions back."""
    def __init__(self):
        from yolox.tracker.byte_tracker import BYTETracker
        observe_detector_boxes()
        self.tracker = BYTETracker(SimpleNamespace(track_thresh=.6,track_buffer=30,
                                  match_thresh=.9,mot20=False),30)
        self.counter = 0; self.frame = 0
        self.boxes = np.empty((0,4), np.float32)
        self.track_hash = hashlib.sha256()

    def advance(self, detections, shape, frame):
        from yolox.tracker.basetrack import BaseTrack
        if frame != self.frame+1: raise ValueError('Nonconsecutive replay')
        saved = BaseTrack._count
        try:
            BaseTrack._count = self.counter
            tracks = self.tracker.update(detections.copy(), shape, shape)
            self.counter = BaseTrack._count
        finally:
            BaseTrack._count = saved
        for t in tracks:
            x,y,w,h = t.feedback_box
            self.track_hash.update(f'{frame},{int(t.track_id)},{x:.1f},{y:.1f},{w:.1f},{h:.1f},{float(t.score):.6f},-1,-1,-1\n'.encode())
        self.boxes = observed_history_xyxy(tracks); self.frame = frame


def detection_lines(output, head, ratio, frame):
    from yolox.utils import postprocess
    raw = decode(output.float().cpu(), head.hw, head.strides)
    output = postprocess(raw, 1, .001, .7)[0]
    if output is None: return ''
    boxes = output[:, :4].numpy().astype(np.float64)/ratio
    xywh = np.c_[boxes[:, :2], boxes[:, 2:]-boxes[:, :2]].round(4)
    # Reproduce online.py's rounding and xyxy round trip exactly.
    boxes = np.c_[xywh[:, :2], xywh[:, :2]+xywh[:, 2:]]
    scores = (output[:,4]*output[:,5]).numpy().astype(np.float64).round(6)
    return ''.join(f'{frame},-1,{x1:.4f},{y1:.4f},{x2-x1:.4f},{y2-y1:.4f},{s:.6f},-1,-1,-1\n'
                   for (x1,y1,x2,y2),s in zip(boxes,scores))


def targets_from_gt(boxes, ratio, device):
    boxes=np.asarray(list(boxes.values()),np.float32).reshape(-1,4)
    targets=torch.zeros((1,max(1,len(boxes)),5),device=device)
    if len(boxes):
        targets[0,:len(boxes),1:3]=torch.from_numpy((boxes[:,:2]+boxes[:,2:]/2)*ratio).to(device)
        targets[0,:len(boxes),3:]=torch.from_numpy(boxes[:,2:]*ratio).to(device)
    return targets


def json_write(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify_tree(path):
    count=0
    for line in (path/'outputs.sha256').read_text().splitlines():
        digest,name=line.split('  ',1); relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe hash path')
        if sha256(path/relative)!=digest: raise ValueError('Parent hash mismatch: '+name)
        count+=1
    return count


@torch.no_grad()
def run(args, cfg, manifest):
    parent_manifest=json.loads((args.parent/'manifest.yaml').read_text())
    parent_metrics=json.loads((args.parent/'metrics.json').read_text())
    if int((args.parent/'exit_code.txt').read_text())!=0 or parent_metrics['final_epoch']!=10:
        raise ValueError('Training has not successfully finished epoch ten')
    if parent_manifest['status']!='completed' or parent_manifest['code_commit']!=cfg['parent_code_commit']:
        raise ValueError('Wrong parent training lineage')
    epoch=args.parent/'epochs/epoch_010'; manifest['verified_parent_files']=verify_tree(args.parent)
    epoch_metrics=json.loads((epoch/'metrics.json').read_text()); adapter=epoch/'adapter.pt'
    if sha256(adapter)!=epoch_metrics['checkpoint_sha256']: raise ValueError('Epoch10 adapter mismatch')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=cfg['gpu_uuid'] or torch.cuda.device_count()!=1:
        raise ValueError('Expected only physical GPU0')
    if cfg['checkpoint_epoch']!=10 or cfg['test_access'] or cfg['modes']!=list(MODES):
        raise ValueError('Counterfactual protocol drift')
    manifest['inputs']=verify_inputs(cfg,args.config,args.data_root,args.checkpoint,'pilot')
    torch.set_num_threads(2);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=FrozenTraDeS(args.checkpoint,input_size=cfg['input_size']).cuda().eval()
    metadata=load_adapter(model,adapter,args.checkpoint)
    if metadata['epoch']!=10: raise ValueError('Wrong adapter metadata')
    before=tensor_hashes(model); head=model.detector.head;head.decode_in_inference=False
    manifest.update(adapter_sha256=sha256(adapter),adapter_path=str(adapter),gpu_name=torch.cuda.get_device_name(0),
                    adapter_metadata=metadata,matched_online_caches={},matched_replayed_tracks={},matched_off_caches={})
    json_write(args.output/'manifest.yaml',manifest)
    data=read_train_sequences(args.data_root,cfg['dev_sequences']); frame_count=paired_count=0
    totals={mode:dict(detection_loss=0.,iou_loss=0.,conf_loss=0.,cls_loss=0.,l1_loss=0.,num_fg=0.) for mode in MODES}
    fields=['sequence','frame','mode','history_frame','gt_history_count','online_history_count','active',
            *totals['off'].keys(),'residual_rms']
    with (args.output/'per_frame.csv').open('w',buffering=1) as log:
        writer=csv.DictWriter(log,fieldnames=fields);writer.writeheader()
        for name,(path,info,frames) in data.items():
            length=int(info['seqlength']); replay=HistoryReplay();previous=None; geometry=None
            reference=cache_detections(epoch/'validation/on'/name/'det.txt',length)
            with contextlib.ExitStack() as stack:
                files={}
                for mode in MODES:
                    dest=args.output/'validation'/mode/name;dest.mkdir(parents=True)
                    files[mode]=stack.enter_context((dest/'det.txt').open('w'))
                for frame in range(1,length+1):
                    image,ratio,shape=read_image(path,frame,info,model.input_size);image=image[None].cuda()
                    current_geometry=(tuple(image.shape),tuple(shape),ratio)
                    if geometry is not None and current_geometry!=geometry:raise ValueError('Geometry drift')
                    geometry=current_geometry
                    features=model.detector.backbone(image);raw=features[0]
                    gt_boxes=np.asarray(list(frames[frame-1].values()) if frame>1 else [],np.float32).reshape(-1,4)
                    gt_boxes[:,2:]+=gt_boxes[:,:2]
                    gt_heat=render_heat(gt_boxes*ratio,model.input_size)[None].cuda()
                    online_heat=render_heat(replay.boxes*ratio,model.input_size)[None].cuda()
                    target=targets_from_gt(frames[frame],ratio,image.device)
                    for mode in MODES:
                        heat=gt_heat if mode=='gt' else online_heat
                        with zero_history_support(model.adapter.core,mode=='zero_support'):
                            refined,cva,residual=model.adapter(raw,previous,heat,enabled=mode!='off')
                        level=(refined,*features[1:]); prediction=head(level)
                        files[mode].write(detection_lines(prediction,head,ratio,frame))
                        try:
                            head.training=True; values=head(level,target,image)
                        finally:head.training=False
                        loss={k:float(v) for k,v in zip(totals[mode],values)}
                        if not all(np.isfinite(v) for v in loss.values()):raise FloatingPointError('Nonfinite diagnostic loss')
                        if frame>1:
                            for key,v in loss.items():totals[mode][key]+=v
                        writer.writerow(dict(sequence=name,frame=frame,mode=mode,history_frame=frame-1,
                            gt_history_count=len(gt_boxes),online_history_count=len(replay.boxes),active=cva is not None,
                            **loss,residual_rms=float(residual.float().square().mean().sqrt())))
                    previous=raw.detach(); replay.advance(reference[frame],shape,frame)
                    frame_count+=1;paired_count+=frame>1
                    if frame%50==0:
                        progress=dict(sequence=name,frame=frame,completed_frames=frame_count,total_frames=1708)
                        json_write(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
            checks=dict(online=sha256(args.output/'validation/online'/name/'det.txt')==sha256(epoch/'validation/on'/name/'det.txt'),
                        off=sha256(args.output/'validation/off'/name/'det.txt')==sha256(epoch/'validation/off'/name/'det.txt'),
                        replay=replay.track_hash.hexdigest()==sha256(epoch/'validation/on'/name/'tracks.txt'))
            manifest['matched_online_caches'][name]=checks['online'];manifest['matched_off_caches'][name]=checks['off']
            manifest['matched_replayed_tracks'][name]=checks['replay'];json_write(args.output/'manifest.yaml',manifest)
            if not all(checks.values()):raise RuntimeError('Paired baseline/replay identity failed: '+str(checks))
    if frame_count!=1708 or paired_count!=1702:raise ValueError('Unexpected dev frame count')
    if tensor_hashes(model)!=before or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Weights, BN or gradients changed during read-only audit')
    results={}
    for mode in MODES:
        cache=args.output/'validation'/mode
        result=evaluate(cache,args.data_root,cfg['dev_sequences'])
        result['per_sequence']=[dict(sequence=name,**evaluate(cache,args.data_root,[name])) for name in cfg['dev_sequences']]
        result['paired_mean_losses']={k:v/paired_count for k,v in totals[mode].items()}
        json_write(cache/'metrics.json',result);results[mode]=result
    pairs=[('gt','online'),('online','zero_support'),('zero_support','off'),('gt','off'),('online','off')]
    contrasts={a+'-minus-'+b:dict(AP50_95_pp=100*(results[a]['metrics']['AP50:95']-results[b]['metrics']['AP50:95']),
        detection_loss=results[a]['paired_mean_losses']['detection_loss']-results[b]['paired_mean_losses']['detection_loss']) for a,b in pairs}
    json_write(args.output/'metrics.json',dict(checkpoint_epoch=10,frames=frame_count,paired_frames=paired_count,
        results=results,contrasts=contrasts,all_weights_and_BN_unchanged=True,optimizer_steps=0,
        detector_seen_dev=True,test_access=False,gt_is_oracle=True,
        caution='Controlled fixed-cache interventions; not separate closed-loop deployable trackers. No retraining attribution.'))
    manifest.update(status='completed',frames=frame_count,paired_frames=paired_count,all_weights_and_BN_unchanged=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','parent','data-root','checkpoint','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'manifest.yaml').exists():raise FileExistsError('Immutable diagnostic attempt exists')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Clean source required')
    cfg=json.loads(args.config.read_text());json_write(args.output/'config.json',cfg)
    manifest=dict(phase='paired_history_counterfactual',status='running',parent_run='run_005',checkpoint_epoch=10,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),optimizer_steps=0,test_access=False,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_dirty=False,
        config_sha256=sha256(args.config),effective_argv=sys.argv,working_directory=os.getcwd(),
        python_executable=sys.executable,python=sys.version,torch=str(torch.__version__),torchvision=str(torchvision.__version__),
        platform=platform.platform(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'))
    json_write(args.output/'manifest.yaml',manifest)
    try:run(args,cfg,manifest)
    except BaseException as error:manifest.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        manifest['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();json_write(args.output/'manifest.yaml',manifest)


if __name__=='__main__':main()
