"""Exp162 CPU preflight / frozen B0 coverage. Never launches formal training."""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import types

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from .data import read_image, sequence_info, read_train_sequences
from .weak import (WeakDetector, WeakOnline, CausalState, DetectionTracker,
                   heat_from_record, postprocess_rows, render, select_rows)
from .weak_history import sha256, train_batch, make_optimizer

REFERENCE = '46820526c6477c3498ae596711df7fc9fb7633f6'


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def reference_tracker_class():
    source = subprocess.check_output(['git','show',REFERENCE+':yolox/tracker/byte_tracker.py'],text=True)
    module = types.ModuleType('yolox.tracker.exp162_reference')
    module.__package__ = 'yolox.tracker'
    exec(compile(source, '<pinned-reference-ByteTrack>', 'exec'), module.__dict__)
    cls = module.STrack
    act, upd, rea = cls.activate, cls.update, cls.re_activate
    def activate(self, kalman, frame):
        self.feedback_box = self._tlwh.copy()
        return act(self,kalman,frame)
    def update(self,new,frame):
        self.feedback_box = new._tlwh.copy()
        return upd(self,new,frame)
    def reactivate(self,new,frame,new_id=False):
        self.feedback_box = new._tlwh.copy()
        return rea(self,new,frame,new_id)
    cls.activate,cls.update,cls.re_activate = activate,update,reactivate
    return module.BYTETracker


def formatted_tracks(frame, tracks):
    return ''.join(f'{frame},{identity},{x:.1f},{y:.1f},{w:.1f},{h:.1f},{s:.6f},-1,-1,-1\n'
                   for identity,x,y,w,h,s in tracks)


def xywh_to_xyxy(boxes):
    boxes=np.asarray(boxes,np.float64).reshape(-1,4).copy()
    boxes[:,2:]+=boxes[:,:2]
    return boxes


def ious(a,b):
    a=np.asarray(a,np.float64).reshape(-1,4);b=np.asarray(b,np.float64).reshape(-1,4)
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.maximum(hi-lo,0).prod(-1)
    return inter/np.maximum((a[:,2:]-a[:,:2]).prod(-1)[:,None]+(b[:,2:]-b[:,:2]).prod(-1)[None,:]-inter,1e-12)


def match_gt(boxes, gt):
    """Max-cardinality >=.5 then IoU; stable row/GT ordering resolves ties."""
    identities=sorted(gt)
    values=ious(boxes,xywh_to_xyxy([gt[i] for i in identities]))
    if not values.size:return {}
    valid=values>=.5
    n=min(values.shape)
    # Invalid pairs have zero utility; n+1 prioritizes cardinality over total IoU.
    utility=valid*(n+1+values)
    a,b=linear_sum_assignment(-utility)
    return {identities[j]:int(i) for i,j in zip(a,b) if valid[i,j]}


def check_inputs(root, protocol, manifest, images=False):
    data=root/protocol['dataset']['root']
    checkpoint=root/protocol['initialization']['checkpoint']
    checksum=sha256(checkpoint)
    if checksum!=protocol['initialization']['checkpoint_sha256']:
        raise ValueError('Checkpoint identity mismatch')
    manifest['input_hashes'][str(checkpoint)]=checksum
    if images:
        for name in protocol['preflight']['sequences']:
            path=data/'train'/name; info=sequence_info(path)
            for file in [path/'seqinfo.ini',path/'gt/gt.txt']:
                manifest['input_hashes'][str(file)]=sha256(file)
            for frame in range(1,21):
                file=path/info.get('imdir','img1')/f"{frame:06d}{info.get('imext','.jpg')}"
                manifest['input_hashes'][str(file)]=sha256(file)
    return data,checkpoint


def coverage(root, protocol, output, manifest):
    data,_=check_inputs(root,protocol,manifest)
    cache=root/'experiments/exp141_bytetrack_yolox_x_uavswarmv1/results/run_006/detections'
    meta_path=cache/'export_summary.json';meta=json.loads(meta_path.read_text())
    manifest['input_hashes'][str(meta_path)]=sha256(meta_path)
    assert meta['checkpoint_sha256']==protocol['initialization']['checkpoint_sha256']
    assert meta['split']=='train' and meta['total_frames']==6844 and not meta['gt_read']
    assert meta['test_size']==[800,1440] and meta['conf']==.001 and meta['nms']==.7
    assert meta['precision']=='fp32' and meta['batch']==1 and meta['tta'] is False
    expected={x['sequence']:x for x in meta['sequences']}
    manifest['cache_origin']={'path':str(cache),'generator_device':meta['device'],
                             'new_detector_inference':False,'reason':'verified_existing_B0_train_cache'}
    names=protocol['dataset']['fit_sequences']+protocol['dataset']['dev_sequences']
    ref_class=reference_tracker_class()
    summaries=[]; all_events=[]; all_scores=[]
    (output/'output').mkdir()
    for name in names:
        path=data/'train'/name;info=sequence_info(path);length=int(info['seqlength'])
        shape=(int(info['imheight']),int(info['imwidth']));ratio=min(800/shape[0],1440/shape[1])
        file=cache/name/'det.txt'
        assert sha256(file)==expected[name]['det_sha256']
        assert expected[name]['frames']==length
        for f in [file,path/'seqinfo.ini']:
            manifest['input_hashes'][str(f)]=sha256(f)
        values=np.loadtxt(file,delimiter=',',ndmin=2)
        assert len(values)==expected[name]['detections'] and np.isfinite(values).all()
        assert np.all((values[:,0]>=1)&(values[:,0]<=length)&(values[:,0]==values[:,0].astype(int)))
        by_frame={t:values[values[:,0]==t] for t in range(1,length+1)}
        on,off,reference=DetectionTracker(True),DetectionTracker(False),DetectionTracker(False,ref_class)
        state=CausalState('TU');state.reset(name)
        records=[]; track_lines=[]; ref_lines=[]; history_lines=[]
        summary={'sequence':name,'split':'dev' if name in protocol['dataset']['dev_sequences'] else 'fit',
                 'frames':length,'detections':len(values),'T':0,'U':0,'low_consumed':0,
                 'high_unconsumed':0,'unconfirmed_consumed':0,'U_fully_covered':0,
                 'incremental_pixels':0,'incremental_frames':0,'usable_history_incremental_frames':0}
        for frame in range(1,length+1):
            values_t=by_frame[frame]
            dets=np.c_[xywh_to_xyxy(values_t[:,2:6]),values_t[:,6]]
            # A training record is captured BEFORE processing the current frame.
            history=state.snapshot(name,frame,protocol['initialization']['checkpoint_sha256'])
            history_lines.append(json.dumps(history)+'\n')
            tracks,audit=on.update(dets,shape)
            tracks_off,_=off.update(dets,shape);tracks_ref,_=reference.update(dets,shape)
            assert tracks==tracks_off==tracks_ref,(name,frame,'tracker differs')
            text=formatted_tracks(frame,tracks);text_ref=formatted_tracks(frame,tracks_ref)
            track_lines.append(text);ref_lines.append(text_ref)
            t,u=select_rows(dets,audit);state.advance(name,frame,dets,audit)
            ht,hu=render(dets[t],ratio),render(dets[u],ratio)
            incremental=int((hu>ht).sum());summary['incremental_pixels']+=incremental
            summary['incremental_frames']+=int(incremental>0)
            summary['usable_history_incremental_frames']+=int(incremental>0 and frame<length)
            summary['T']+=len(t);summary['U']+=len(u)
            for i in u:
                summary['U_fully_covered']+=int(not torch.any(render(dets[[i]],ratio)>ht))
                all_scores.append(float(dets[i,4]))
            summary['low_consumed']+=int(sum(.1<d[4]<.6 and a['consumed_by'] is not None for d,a in zip(dets,audit)))
            summary['high_unconsumed']+=int(sum(d[4]>=.6 and a['consumed_by'] is None for d,a in zip(dets,audit)))
            summary['unconfirmed_consumed']+=int(sum(a['consumed_by'] is not None and not a['output_confirmed'] for a in audit))
            records.append({'frame':frame,'detections':dets.tolist(),'audit':audit,'t':t,'u':u,
                            'tracks':tracks,'incremental_pixels':incremental})
        dest=output/'output'/name;dest.mkdir()
        (dest/'tracks.txt').write_text(''.join(track_lines))
        (dest/'reference_tracks.txt').write_text(''.join(ref_lines))
        (dest/'history.jsonl').write_text(''.join(history_lines))
        (dest/'ownership.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        assert sha256(dest/'tracks.txt')==sha256(dest/'reference_tracks.txt')
        summary['byte_identical_reference']=True
        # The detector/tracker/feedback stage has ended; only NOW read annotations.
        if summary['split']=='dev':
            manifest['input_hashes'][str(path/'gt/gt.txt')]=sha256(path/'gt/gt.txt')
            gt=read_train_sequences(data,[name])[name][2]
            matches=[]; weak_matches=[]
            for r in records:
                g=gt[r['frame']]
                matches.append(match_gt(xywh_to_xyxy([x[1:5] for x in r['tracks']]),g))
                d=np.asarray(r['detections']).reshape(-1,5)
                weak_matches.append(match_gt(d[r['u'],:4],g))
            events=[]; displacement=[]
            for i in range(1,length):
                for identity,local_row in weak_matches[i-1].items():
                    if identity not in gt[i+1]:continue
                    r=records[i-1];row_id=r['u'][local_row]
                    box=np.asarray(r['detections'][row_id])
                    center=(box[:2]+box[2:4])*.5
                    next_gt=gt[i+1][identity];next_center=next_gt[:2]+next_gt[2:]*.5
                    displacement.append(float(np.linalg.norm(next_center-center)*ratio/8))
                    if identity not in matches[i-1] and identity not in matches[i]:
                        events.append({'sequence':name,'frame':i+1,'gt_id':identity,
                                       'previous_det_row_id':row_id,'previous_score':float(box[4])})
            summary['events']=len(events)
            summary['weak_to_next_GT_center_grid_distance_quantiles']=np.quantile(displacement,[.5,.9,.99]).tolist() if displacement else []
            all_events.extend(events)
        summaries.append(summary)
        print(json.dumps(summary),flush=True)
    fit=[s for s in summaries if s['split']=='fit'];dev=[s for s in summaries if s['split']=='dev']
    fit_has=sum(s['usable_history_incremental_frames'] for s in fit)>0
    event_sequences=sum(s['events']>0 for s in dev)
    result={'per_sequence':summaries,'cache_generator_device':meta['device'],
            'coverage_device':'cpu','new_detector_inference':False,
            'all_reference_tracks_byte_identical':all(s['byte_identical_reference'] for s in summaries),
            'frames':sum(s['frames'] for s in summaries),'detections':sum(s['detections'] for s in summaries),
            'fit':{k:sum(s[k] for s in fit) for k in ['T','U','low_consumed','U_fully_covered','usable_history_incremental_frames']},
            'dev':{k:sum(s[k] for s in dev) for k in ['T','U','low_consumed','U_fully_covered','usable_history_incremental_frames','events']},
            'event_sequences':event_sequences,'score_quantiles':np.quantile(all_scores,[0,.25,.5,.75,1]).tolist() if all_scores else [],
            'gate':{'fit_nonzero_incremental':fit_has,'events_min_30':len(all_events)>=30,
                    'event_sequences_min_3':event_sequences>=3},
            'formal_training_started':False}
    result['passed']=all(result['gate'].values())
    dump(output/'events.json',all_events)
    return result


def preflight(root,protocol,output,manifest):
    data,checkpoint=check_inputs(root,protocol,manifest,images=True)
    torch.manual_seed(0)
    model=WeakDetector(checkpoint).eval()
    model.detector.head.decode_in_inference=False
    runner=WeakOnline(model,'TU',sha256(checkpoint))
    samples=[]; frames=[]
    for name in protocol['preflight']['sequences']:
        path=data/'train'/name;info=sequence_info(path);runner.reset(name)
        ref=DetectionTracker(False,reference_tracker_class())
        for frame in range(1,21):
            image,ratio,shape=read_image(path,frame,info,(800,1440));image=image[None]
            history=runner.state.snapshot(name,frame,sha256(checkpoint) if frame==1 else runner.snapshot_sha)
            h=heat_from_record(history,ratio,(100,180))[None]
            with torch.no_grad():
                # The direct original YOLOX call is independent of residual injection.
                raw_b0=model.detector(image)
                raw_off=model(image,h,enabled=False)
                raw_zero=model(image,h)
            assert torch.equal(raw_b0,raw_off) and torch.equal(raw_b0,raw_zero),(name,frame)
            dets=postprocess_rows(raw_zero,model.detector.head,ratio)
            baseline_dets=postprocess_rows(raw_b0,model.detector.head,ratio)
            np.testing.assert_array_equal(dets,baseline_dets)
            tracks,audit=runner.tracker.update(dets,shape)
            reference,_=ref.update(baseline_dets,shape)
            assert formatted_tracks(frame,tracks)==formatted_tracks(frame,reference)
            runner.state.advance(name,frame,dets,audit)
            frames.append({'sequence':name,'frame':frame,'raw_equal':True,'detections_equal':True,
                           'tracks_equal':True,'history_source_frame':history['source_frame'],
                           'T':len(runner.state.t),'U':len(runner.state.u),'heat_nonzero':bool(h.any())})
            if frame in protocol['preflight']['loss_probe_frames']:
                samples.append((name,frame,image,h,ratio))
            if frame%5==0:print(f'identity {name} {frame}/20',flush=True)
    dump(output/'identity.json',frames)
    # GT is opened only after all image-only causal histories have been produced.
    annotations=read_train_sequences(data,protocol['preflight']['sequences'])
    batches=[]
    for start in (0,2):
        part=samples[start:start+2];target_list=[]
        for name,frame,image,h,ratio in part:
            boxes=np.asarray(list(annotations[name][2][frame].values()),np.float32).reshape(-1,4)
            target=torch.zeros((len(boxes),5));target[:,1:3]=torch.from_numpy((boxes[:,:2]+boxes[:,2:]*.5)*ratio)
            target[:,3:]=torch.from_numpy(boxes[:,2:]*ratio);target_list.append(target)
        target=torch.zeros(2,max(1,max(len(t) for t in target_list)),5)
        for i,t in enumerate(target_list):target[i,:len(t)]=t
        batches.append((torch.cat([x[2] for x in part]),torch.cat([x[3] for x in part]),target))
    optimizer=make_optimizer(model)
    before=[p.detach().clone() for p in model.adapter.parameters()]
    losses=[];gradients=[]
    # Four updates suffice to exercise the zero-initialized output path; cap remains 20.
    for step in range(4):
        loss=train_batch(model,optimizer,*batches[step%2]);losses.append(loss)
        gradients.append([float(p.grad.norm()) for p in model.adapter.parameters()])
        print(json.dumps({'probe_step':step+1,'loss':loss,'adapter_grad_norm':gradients[-1]}),flush=True)
    changes=[float((p-old).norm()) for p,old in zip(model.adapter.parameters(),before)]
    assert all(x>0 for x in changes)
    assert any(x[0]>0 for x in gradients[1:]) and any(x[1]>0 for x in gradients)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    model.eval();image,heat,_=batches[0]
    with torch.no_grad():
        direct=model.detector(image)
        off=model(image,heat,enabled=False)
        zero=model(image,torch.zeros_like(heat))
        on,residual=model(image,heat,return_residual=True)
    assert torch.equal(direct,off) and torch.equal(direct,zero)
    assert float(residual.abs().max())>0 and not torch.equal(on,off)
    saved=output/'diagnostic_snapshot.pt'
    torch.save({'weak_config':'exp162_v1','model':model.state_dict(),'epoch':0,'preflight':True},saved)
    snapshot_hash=sha256(saved)
    # Reload without creating another full model: overwrite, then restore exact state.
    with torch.no_grad():model.adapter.output.weight.zero_()
    model.load_state_dict(torch.load(saved,map_location='cpu',weights_only=True)['model'],strict=True)
    with torch.no_grad():reloaded=model(image,heat)
    assert torch.equal(on,reloaded)
    return {'passed':True,'identity_frames':len(frames),'sequence_reset_count':2,
            'all_raw_detections_tracks_equal':True,'nonzero_heat_frames':sum(x['heat_nonzero'] for x in frames),
            'bounded_optimizer_steps':4,'adapter_gradient_norms':gradients,'adapter_weight_change_norms':changes,
            'losses':losses,'trained_weights_disabled_vs_same_weights_original_equal':True,
            'trained_weights_zero_heat_vs_same_weights_original_equal':True,
            'nonzero_feedback_changes_raw':True,'residual_max_abs':float(residual.abs().max()),
            'checkpoint_reload_equal':True,'diagnostic_snapshot_sha256':snapshot_hash,
            'diagnostic_weights_for_formal_initialization':False,'formal_training_started':False,
            'no_claim_trained_detector_equals_initial_B0':True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['preflight','coverage'],required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if git('status','--porcelain'):raise RuntimeError('Clean committed source required')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    manifest={'phase':args.phase,'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'code_commit':git('rev-parse','HEAD'),'code_dirty':False,'branch':git('branch','--show-current'),
              'private_repository':'git@github.com:YUSIO/ByteTrack-MOT.git','reference_commit':REFERENCE,
              'protocol_sha256':sha256(args.protocol),'python_executable':sys.executable,'python':sys.version,
              'torch':str(torch.__version__),'numpy':np.__version__,'host':platform.node(),
              'device':'cpu','threads':4,'seed':0,'formal_training_authorized':False,
              'input_hashes':{},'command':sys.argv,'artifact_recovery':'pending'}
    protocol=json.loads(args.protocol.read_text());dump(args.output/'frozen_protocol.json',protocol)
    import shlex
    (args.output/'command.txt').write_text(shlex.join([sys.executable,'-m','feedback.weak_check',*sys.argv[1:]])+'\n')
    dump(args.output/'manifest.yaml',manifest);started=time.time();rc=1
    try:
        result=(preflight if args.phase=='preflight' else coverage)(args.root,protocol,args.output,manifest)
        dump(args.output/'metrics.json',result)
        # Gate-negative coverage is a completed measurement, not an execution failure.
        manifest['status']='completed';rc=0
    except Exception:
        manifest['status']='failed';(args.output/'exception.txt').write_text(traceback.format_exc());raise
    finally:
        manifest.update(seconds=time.time()-started,exit_code=rc,
                        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        dump(args.output/'manifest.yaml',manifest);(args.output/'exit_code.txt').write_text(str(rc)+'\n')
        files=sorted(p for p in args.output.rglob('*') if p.is_file() and p.name!='output_checksums.sha256')
        (args.output/'output_checksums.sha256').write_text(''.join(f'{sha256(p)}  {p.relative_to(args.output)}\n' for p in files))
    return rc


if __name__=='__main__':raise SystemExit(main())
