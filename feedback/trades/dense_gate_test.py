"""Frozen epoch10 test export, followed by GT-only scoring of closed caches."""
import argparse,contextlib,datetime,hashlib,io,json,os,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import torchvision
from feedback.data import sequence_info,read_image
from .checkpoint import load_adapter,sha256
from .model import FrozenTraDeS
from .data import render_heat
from .preflight import tensor_hashes
from .history_counterfactual import HistoryReplay,detection_lines
from .dense_gate import geometry,recipient_mask,residuals,blend
from .validate import NAMES

MODES=('OFF','A','Mr_shared','Mr_closed')

def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def from_lines(lines):
    rows=[]
    for line in lines.splitlines():
        v=list(map(float,line.split(',')));x,y,w,h=v[2:6];rows.append([x,y,x+w,y+h,v[6]])
    return np.asarray(rows,np.float64).reshape(-1,5)

def score_cache(cache,gtroot,sequences):
    """Exp141/Exp088 COCO semantics, plus original score-.6/IoU-.5 FP/FN."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    images,annotations,detections=[],[],[];seqids={};gtcounts={}
    for name in sequences:
        path=gtroot/name;info=sequence_info(path)
        width,height,length=[int(info[k]) for k in ('imwidth','imheight','seqlength')]
        base=len(images);seqids[name]=list(range(base+1,base+length+1));start=len(annotations)
        images.extend(dict(id=base+f,width=width,height=height) for f in range(1,length+1))
        for row in np.loadtxt(path/'gt/gt.txt',delimiter=',',ndmin=2):
            frame,_,x,y,w,h=row[:6]
            assert 1<=int(frame)<=length and row[6]>=1
            x1,y1,x2,y2=max(x,0),max(y,0),min(x+w,width),min(y+h,height)
            if x2>x1 and y2>y1:
                annotations.append(dict(id=len(annotations)+1,image_id=base+int(frame),category_id=1,iscrowd=0,bbox=[x1,y1,x2-x1,y2-y1],area=(x2-x1)*(y2-y1)))
        gtcounts[name]=len(annotations)-start
        for line in (cache/name/'det.txt').read_text().splitlines():
            v=list(map(float,line.split(',')));assert 1<=int(v[0])<=length
            detections.append(dict(image_id=base+int(v[0]),category_id=1,bbox=v[2:6],score=v[6]))
    def result(ev,count):
        tp=fp=0
        for item in ev.evalImgs:
            if item is None or item['aRng']!=ev.params.areaRng[0]:continue
            sel=(np.asarray(item['dtScores'])>=.6)&~item['dtIgnore'][0]
            tp+=int(((item['dtMatches'][0]>0)&sel).sum());fp+=int(((item['dtMatches'][0]==0)&sel).sum())
        return dict(metrics=dict(zip(NAMES,map(float,ev.stats))),fixed_iou50_score06=dict(TP=tp,FP=fp,FN=count-tp),gt_boxes=count)
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO();gt.dataset=dict(images=images,annotations=annotations,categories=[dict(id=1,name='UAV')]);gt.createIndex()
        if detections:dt=gt.loadRes(detections)
        else:
            dt=COCO();dt.dataset=dict(images=images,annotations=[],categories=gt.dataset['categories']);dt.createIndex()
        ev=COCOeval(gt,dt,'bbox');ev.evaluate();ev.accumulate();ev.summarize()
        output=result(ev,len(annotations));output.update(frames=len(images),detections=len(detections),per_sequence={})
        for name,ids in seqids.items():
            e=COCOeval(gt,dt,'bbox');e.params.imgIds=ids;e.evaluate();e.accumulate();e.summarize()
            output['per_sequence'][name]=result(e,gtcounts[name]);output['per_sequence'][name]['frames']=len(ids)
    assert len(images)==5754 and len(annotations)==37664
    return output


@torch.no_grad()
def run(args,cfg,manifest):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==cfg['gpu_uuid'] and torch.cuda.device_count()==1
    assert cfg['modes']==list(MODES)
    assert cfg['sequences']==[f'UAVSwarm-{i:02d}' for i in range(2,73,2)]
    assert sha256(args.root/cfg['adapter'])==cfg['adapter_sha256']
    assert sha256(args.root/cfg['base'])==cfg['base_sha256']
    torch.set_num_threads(2);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=FrozenTraDeS(args.root/cfg['base']).cuda().eval().requires_grad_(False)
    metadata=load_adapter(model,args.root/cfg['adapter'],args.root/cfg['base'])
    assert metadata['epoch']==10 and model.adapter.fusion_mode=='visual_only'
    model.detector.head.decode_in_inference=False
    before=tensor_hashes(model);count=0;started=time.time();sequence_summaries=[]
    def predict(features,ratio,frame):
        lines=detection_lines(model.detector.head(features),model.detector.head,ratio,frame)
        det=from_lines(lines);assert np.isfinite(det).all()
        return lines,det
    with (args.output/'gate_trace.jsonl').open('w',buffering=1) as trace,(args.output/'image_inputs.sha256').open('w',buffering=1) as inputlog:
        for name in cfg['sequences']:
            # Inference reads only images and sequence geometry. No GT loader.
            path=args.root/'datasets/UAVSwarm-dataset-master/test'/name;info=sequence_info(path)
            assert sha256(path/'seqinfo.ini')==cfg['test_seqinfo_sha256'][name]
            history_a,history_g=HistoryReplay(),HistoryReplay()
            previous=None;used_a=used_g=0;gate_sum=0.;geo_shape=None
            with contextlib.ExitStack() as stack:
                files={}
                for mode in MODES:
                    folder=args.output/'detections'/mode/name;folder.mkdir(parents=True)
                    files[mode]=stack.enter_context((folder/'det.txt').open('w'))
                for frame in range(1,int(info['seqlength'])+1):
                    filename=path/info.get('imdir','img1')/f"{frame:06d}{info.get('imext','.jpg')}"
                    inputlog.write(sha256(filename)+'  '+str(filename.relative_to(args.root))+'\n')
                    image,ratio,shape=read_image(path,frame,info,model.input_size)
                    valid=(int(shape[0]*ratio),int(shape[1]*ratio))
                    if geo_shape is not None:assert geo_shape==(ratio,shape)
                    geo_shape=(ratio,shape)
                    features=model.detector.backbone(image[None].cuda());raw=features[0]
                    off_lines,off_det=predict(features,ratio,frame)
                    files['OFF'].write(off_lines)
                    m=recipient_mask(off_det,ratio,raw.shape[-2:],valid,raw.device)
                    ha=render_heat(history_a.boxes*ratio,model.input_size)[None].cuda()
                    hg=render_heat(history_g.boxes*ratio,model.input_size)[None].cuda()
                    active_a=previous is not None and bool(ha.max()>0)
                    active_g=previous is not None and bool(hg.max()>0)
                    gate=raw[:,:1]*0;stats={};va=vd=vg_a=vg_d=None
                    if active_a or active_g:
                        geo=geometry(model.adapter,raw,previous,valid)
                        gate=m*geo['reliability'];stats=geo['stats']
                        assert bool(torch.isfinite(gate).all()) and float(gate.min())>=0 and float(gate.max())<=1
                        if active_a:
                            va,vd,attention_a=residuals(model.adapter,geo,ha)
                            refined_a=raw+va;refined_shared=blend(raw,va,vd,gate)
                        else:refined_a=refined_shared=raw
                        if active_g:
                            if torch.equal(ha,hg) and active_a:vg_a,vg_d,attention_g=va,vd,attention_a
                            else:vg_a,vg_d,attention_g=residuals(model.adapter,geo,hg)
                            refined_closed=blend(raw,vg_a,vg_d,gate)
                        else:refined_closed=raw
                    else:refined_a=refined_shared=refined_closed=raw
                    exports={}
                    for mode,refined in [('A',refined_a),('Mr_shared',refined_shared),('Mr_closed',refined_closed)]:
                        lines,det=predict((refined,*features[1:]),ratio,frame)
                        files[mode].write(lines);exports[mode]=det
                    used_a+=int(active_a);used_g+=int(active_g)
                    trace.write(json.dumps(dict(sequence=name,frame=frame,history_frame=frame-1,active_a=active_a,active_closed=active_g,
                        native_history_count=len(history_a.boxes),closed_history_count=len(history_g.boxes),
                        native_heat_sha256=hashlib.sha256(ha.cpu().numpy().tobytes()).hexdigest(),
                        closed_heat_sha256=hashlib.sha256(hg.cpu().numpy().tobytes()).hexdigest(),
                        gate_mean=float(gate.mean()),gate_max=float(gate.max()),mask_mean=float(m.mean()),
                        closed_minus_native_relative_rms=float((refined_closed-refined_a).square().mean().sqrt()/raw.square().mean().sqrt()),**stats))+'\n')
                    gate_sum+=float(gate.mean())
                    history_a.advance(exports['A'],shape,frame)
                    history_g.advance(exports['Mr_closed'],shape,frame)
                    previous=raw.detach();count+=1
                    if count%50==0:
                        progress=dict(phase='inference',sequence=name,frame=frame,completed_frames=count,total_frames=5754,elapsed_seconds=time.time()-started)
                        dump(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
            sequence_summaries.append(dict(sequence=name,frames=int(info['seqlength']),feedback_frames_A=used_a,feedback_frames_closed=used_g,
                mean_gate=gate_sum/int(info['seqlength']),A_history_tracks_sha256=history_a.track_hash.hexdigest(),closed_history_tracks_sha256=history_g.track_hash.hexdigest()))
            dump(args.output/'inference_sequences.json',sequence_summaries)
    assert count==5754 and before==tensor_hashes(model)
    assert all(p.grad is None for p in model.parameters())
    assert sha256(args.root/cfg['adapter'])==cfg['adapter_sha256']
    manifest.update(inference_completed=True,frames=count,weights_BN_unchanged=True,inference_seconds=time.time()-started)
    dump(args.output/'manifest.yaml',manifest)
    # Ground truth is opened only after every detection cache is finalized.
    gtroot=args.root/cfg['gt_root']
    for name,digest in cfg['canonical_gt_sha256'].items():assert sha256(gtroot/name/'gt/gt.txt')==digest
    results={}
    for mode in MODES:
        print(json.dumps(dict(phase='evaluation',mode=mode)),flush=True)
        dump(args.output/'progress.json',dict(phase='evaluation',mode=mode,completed_frames=count,total_frames=5754))
        result=score_cache(args.output/'detections'/mode,gtroot,cfg['sequences'])
        dump(args.output/'detections'/mode/'metrics.json',result);results[mode]=result
    dump(args.output/'metrics.json',dict(results=results,frames=5754,sequences=36,gt_boxes=37664,
        checkpoint_epoch=10,optimizer_steps=0,weights_BN_unchanged=True,
        history='A causal own predictions; Mr_shared reuses A heat and native attention; Mr_closed uses own causal tracker observations and native A attention under its own history; current M always frozen OFF predictions',
        test_gt_usage='scoring_only_after_inference',no_test_tuning=True))
    manifest.update(status='completed',elapsed_seconds=time.time()-started,test_gt_usage='scoring_only_after_inference')


def main():
    p=argparse.ArgumentParser()
    for key in ('root','config','output'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'manifest.yaml').exists():raise FileExistsError('Existing attempt')
    cfg=json.loads(args.config.read_text());revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert revision==cfg['code_commit'] and not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    manifest=dict(status='running',phase='test_eval',code_commit=revision,code_dirty=False,
        config_sha256=sha256(args.config),adapter_sha256=cfg['adapter_sha256'],base_sha256=cfg['base_sha256'],
        python_executable=sys.executable,python=sys.version,torch=str(torch.__version__),torchvision=str(torchvision.__version__),
        platform=platform.platform(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),effective_argv=sys.argv,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),optimizer_steps=0,test_authorized=True)
    dump(args.output/'manifest.yaml',manifest);dump(args.output/'config.json',cfg)
    try:run(args,cfg,manifest)
    except BaseException as e:manifest.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:manifest['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();dump(args.output/'manifest.yaml',manifest)


if __name__=='__main__':main()
