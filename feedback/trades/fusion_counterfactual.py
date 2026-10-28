"""Fixed epoch10 fusion-term interventions with frozen online history and attention."""
import argparse, contextlib, csv, datetime, hashlib, json, os, platform, subprocess, sys
from pathlib import Path
import numpy as np
import torch
import torchvision
from feedback.data import read_train_sequences, read_image
from .checkpoint import load_adapter, sha256
from .data import render_heat
from .fusion_terms import MODES, frozen_fusion_terms
from .history_counterfactual import HistoryReplay, cache_detections, detection_lines, targets_from_gt, verify_tree, json_write
from .model import FrozenTraDeS
from .preflight import tensor_hashes
from .train import verify_inputs
from .validate import evaluate


@torch.no_grad()
def run(args,cfg,manifest):
    if cfg['checkpoint_epoch']!=10 or cfg['test_access'] or cfg['modes']!=list(MODES):
        raise ValueError('Frozen protocol drift')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=cfg['gpu_uuid'] or torch.cuda.device_count()!=1:
        raise ValueError('Only physical GPU0 allowed')
    pm=json.loads((args.parent/'manifest.yaml').read_text())
    rm=json.loads((args.reference/'manifest.yaml').read_text())
    if pm['status']!='completed' or pm['code_commit']!=cfg['parent_code_commit']:
        raise ValueError('Wrong training parent')
    if rm['status']!='completed' or rm['adapter_sha256']!=cfg['adapter_sha256']:
        raise ValueError('Wrong historical reference')
    for folder in (args.parent,args.reference):
        if int((folder/'exit_code.txt').read_text())!=0:raise ValueError('Parent/reference failed')
    manifest['verified_parent_files']=verify_tree(args.parent)
    manifest['verified_reference_files']=verify_tree(args.reference)
    epoch=args.parent/'epochs/epoch_010';adapter=epoch/'adapter.pt'
    if sha256(adapter)!=cfg['adapter_sha256']:raise ValueError('Wrong epoch10 adapter')
    manifest['inputs']=verify_inputs(cfg,args.config,args.data_root,args.checkpoint,'pilot')
    torch.set_num_threads(2);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=FrozenTraDeS(args.checkpoint,input_size=cfg['input_size']).cuda().eval()
    metadata=load_adapter(model,adapter,args.checkpoint)
    if metadata['epoch']!=10:raise ValueError('Wrong checkpoint metadata')
    model.requires_grad_(False);before=tensor_hashes(model)
    head=model.detector.head;head.decode_in_inference=False
    manifest.update(adapter_sha256=sha256(adapter),gpu_name=torch.cuda.get_device_name(0),matched_full_caches={},
                    matched_off_caches={},matched_replayed_tracks={},matched_run006_caches={})
    json_write(args.output/'manifest.yaml',manifest)
    totals={mode:dict(detection_loss=0.,iou_loss=0.,conf_loss=0.,cls_loss=0.,l1_loss=0.,num_fg=0.) for mode in MODES}
    fields=['sequence','frame','mode','history_frame','online_history_count','active',*totals['off'],'residual_rms']
    frame_count=paired_count=attention_count=0;max_algebra=max_native=0.
    data=read_train_sequences(args.data_root,cfg['dev_sequences'])
    with (args.output/'per_frame.csv').open('w',buffering=1) as log,(args.output/'fusion_audit.jsonl').open('w',buffering=1) as audit_log:
        writer=csv.DictWriter(log,fieldnames=fields);writer.writeheader()
        for name,(path,info,frames) in data.items():
            length=int(info['seqlength']);previous=None;geometry=None;replay=HistoryReplay()
            reference=cache_detections(epoch/'validation/on'/name/'det.txt',length)
            with contextlib.ExitStack() as stack:
                files={}
                for mode in MODES:
                    dest=args.output/'validation'/mode/name;dest.mkdir(parents=True)
                    files[mode]=stack.enter_context((dest/'det.txt').open('w'))
                for frame in range(1,length+1):
                    image,ratio,shape=read_image(path,frame,info,model.input_size);image=image[None].cuda()
                    geo=(tuple(image.shape),tuple(shape),ratio)
                    if geometry is not None and geo!=geometry:raise ValueError('Geometry drift')
                    geometry=geo;features=model.detector.backbone(image);raw=features[0]
                    heat=render_heat(replay.boxes*ratio,model.input_size)[None].cuda()
                    levels,audit,a=frozen_fusion_terms(model.adapter,raw,previous,heat)
                    attention_count+=audit['attention_evaluations'];max_algebra=max(max_algebra,audit['decomposition_max_abs_error'])
                    max_native=max(max_native,audit['native_reconstruction_max_abs_error'])
                    audit_log.write(json.dumps(dict(sequence=name,frame=frame,
                        attention_sha256=hashlib.sha256(a.cpu().numpy().tobytes()).hexdigest() if a is not None else None,
                        online_heat_sha256=hashlib.sha256(heat.cpu().numpy().tobytes()).hexdigest(),**audit))+'\n')
                    target=targets_from_gt(frames[frame],ratio,image.device)
                    for mode in MODES:
                        refined=levels[mode];level=(refined,*features[1:]);prediction=head(level)
                        files[mode].write(detection_lines(prediction,head,ratio,frame))
                        try:head.training=True;values=head(level,target,image)
                        finally:head.training=False
                        losses={k:float(v) for k,v in zip(totals[mode],values)}
                        if not all(np.isfinite(v) for v in losses.values()):raise FloatingPointError('Nonfinite loss')
                        if frame>1:
                            for k,v in losses.items():totals[mode][k]+=v
                        writer.writerow(dict(sequence=name,frame=frame,mode=mode,history_frame=frame-1,
                            online_history_count=len(replay.boxes),active=audit['active'] and mode!='off',**losses,
                            residual_rms=float((refined-raw).square().mean().sqrt())))
                    previous=raw.detach();replay.advance(reference[frame],shape,frame)
                    frame_count+=1;paired_count+=frame>1
                    if frame%50==0:
                        progress=dict(sequence=name,frame=frame,completed_frames=frame_count,total_frames=cfg['expected_frames'])
                        json_write(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
            checks=dict(full=sha256(args.output/'validation/full'/name/'det.txt')==sha256(epoch/'validation/on'/name/'det.txt'),
                        off=sha256(args.output/'validation/off'/name/'det.txt')==sha256(epoch/'validation/off'/name/'det.txt'),
                        replay=replay.track_hash.hexdigest()==sha256(epoch/'validation/on'/name/'tracks.txt'),
                        run006_full=sha256(args.output/'validation/full'/name/'det.txt')==sha256(args.reference/'validation/online'/name/'det.txt'),
                        run006_off=sha256(args.output/'validation/off'/name/'det.txt')==sha256(args.reference/'validation/off'/name/'det.txt'))
            manifest['matched_full_caches'][name]=checks['full'];manifest['matched_off_caches'][name]=checks['off']
            manifest['matched_replayed_tracks'][name]=checks['replay'];manifest['matched_run006_caches'][name]=checks['run006_full'] and checks['run006_off']
            json_write(args.output/'manifest.yaml',manifest)
            if not all(checks.values()):raise RuntimeError('Frozen baseline identity mismatch: '+str(checks))
            print(json.dumps(dict(sequence=name,identity_checks=checks)),flush=True)
    if (frame_count,paired_count)!=(cfg['expected_frames'],cfg['expected_paired_frames']):raise ValueError('Frame count mismatch')
    if tensor_hashes(model)!=before or any(p.grad is not None for p in model.parameters()):raise RuntimeError('Weights/BN/gradients changed')
    results={}
    for mode in MODES:
        cache=args.output/'validation'/mode;result=evaluate(cache,args.data_root,cfg['dev_sequences'])
        result['per_sequence']=[dict(sequence=name,**evaluate(cache,args.data_root,[name])) for name in cfg['dev_sequences']]
        result['paired_mean_losses']={k:v/paired_count for k,v in totals[mode].items()}
        json_write(cache/'metrics.json',result);results[mode]=result
    reference_results=json.loads((args.reference/'metrics.json').read_text())['results']
    for mode,old_mode in [('off','off'),('full','online')]:
        if results[mode]['metrics']!=reference_results[old_mode]['metrics']:raise RuntimeError('Reference metric mismatch')
    pairs=[('no_current','full'),('no_visual','full'),('visual_only','no_current'),('visual_only','off'),('no_current','off'),('no_visual','off'),('full','off')]
    contrasts={a+'-minus-'+b:dict(AP50_95_pp=100*(results[a]['metrics']['AP50:95']-results[b]['metrics']['AP50:95']),
        detection_loss=results[a]['paired_mean_losses']['detection_loss']-results[b]['paired_mean_losses']['detection_loss']) for a,b in pairs}
    output=dict(checkpoint_epoch=10,frames=frame_count,paired_frames=paired_count,results=results,contrasts=contrasts,
        attention_source='native full forward once per active frame; fixed for all term interventions',
        attention_evaluations=attention_count,max_decomposition_abs_error=max_algebra,max_native_reconstruction_abs_error=max_native,
        all_weights_and_BN_unchanged=True,optimizer_steps=0,detector_seen_dev=True,test_access=False,
        caution='Post-hoc frozen-intermediate interventions with old weights; no retraining or additive AP attribution.')
    json_write(args.output/'metrics.json',output)
    manifest.update(status='completed',frames=frame_count,paired_frames=paired_count,all_weights_and_BN_unchanged=True,
                    attention_evaluations=attention_count,max_decomposition_abs_error=max_algebra,max_native_reconstruction_abs_error=max_native)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','parent','reference','data-root','checkpoint','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'manifest.yaml').exists():raise FileExistsError('Attempt already exists')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Clean source required')
    cfg=json.loads(args.config.read_text());json_write(args.output/'config.json',cfg)
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if source!=cfg['code_commit']:raise ValueError('Pinned source commit mismatch')
    manifest=dict(phase='fixed_attention_fusion_counterfactual',status='running',parent_run='run_005',history_reference='run_006',checkpoint_epoch=10,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),optimizer_steps=0,test_access=False,code_commit=source,code_dirty=False,
        config_sha256=sha256(args.config),effective_argv=sys.argv,working_directory=os.getcwd(),python_executable=sys.executable,
        python=sys.version,torch=str(torch.__version__),torchvision=str(torchvision.__version__),platform=platform.platform(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'))
    json_write(args.output/'manifest.yaml',manifest)
    try:run(args,cfg,manifest)
    except BaseException as error:manifest.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:manifest['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();json_write(args.output/'manifest.yaml',manifest)

if __name__=='__main__':main()
