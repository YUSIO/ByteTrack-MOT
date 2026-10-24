"""Train-only two-scalar output geometry calibration. No tracker state mutation."""
from pathlib import Path
import argparse,contextlib,csv,datetime,hashlib,json,platform,shutil,subprocess,sys,traceback
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment,minimize
from homa.evaluate import evaluate

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def scale_boxes(boxes,scales):
    b=np.asarray(boxes,float).copy();center=b[:,:2]+b[:,2:]/2;b[:,2:]*=np.asarray(scales);b[:,:2]=center-b[:,2:]/2;return b

def paired_iou(a,b):
    inter=np.maximum(np.minimum(a[:,:2]+a[:,2:],b[:,:2]+b[:,2:])-np.maximum(a[:,:2],b[:,:2]),0).prod(1)
    return inter/(a[:,2:].prod(1)+b[:,2:].prod(1)-inter)
def matrix_iou(a,b):
    inter=np.maximum(np.minimum(a[:,None,:2]+a[:,None,2:],b[None,:,:2]+b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]),0).prod(2)
    return inter/(a[:,None,2:].prod(2)+b[None,:,2:].prod(2)-inter)
def read(p):
    return np.loadtxt(p,delimiter=',',ndmin=2) if p.stat().st_size else np.empty((0,10))

def fit(B,E,R,cfg,split,inputs):
    pairs=[];stats={}
    for name in split['fit_sequences']:
        tp=B/cfg['train_tracks']/f'{name}.txt';gp=B/cfg['dataset']/'train'/name/'gt/gt.txt';inputs[str(tp.relative_to(B))]=sha(tp);inputs[str(gp.relative_to(B))]=sha(gp)
        tracks=read(tp);gt=read(gp);gt=gt[gt[:,6]>=1];aa=[];bb=[];identity=[]
        for f in sorted(set(tracks[:,0])):
            t=tracks[tracks[:,0]==f];g=gt[gt[:,0]==f]
            if not len(g):continue
            ov=matrix_iou(t[:,2:6],g[:,2:6]);cost=np.where(ov>=cfg['fit_match_iou'],1-ov,1e6);i,j=linear_sum_assignment(cost);keep=ov[i,j]>=cfg['fit_match_iou'];i=i[keep];j=j[keep]
            aa.extend(t[i,2:6]);bb.extend(g[j,2:6]);identity.extend(zip([int(f)]*len(i),t[i,1].astype(int).tolist(),g[j,1].astype(int).tolist()))
        a=np.asarray(aa);b=np.asarray(bb);assert len(a)>0;np.savez_compressed(R/f'{name}.fit_pairs.npz',original=a,target=b,identity=np.array(identity));pairs.append((a,b));stats[name]={'rows':len(tracks),'matched_pairs':len(a),'original_mean_iou':float(paired_iou(a,b).mean())}
    # Equal sequence weight; correspondence is fixed before fitting.
    trace=[]
    def objective(logscale):
        scales=np.exp(logscale);value=float(np.mean([paired_iou(scale_boxes(a,scales),b).mean() for a,b in pairs]));trace.append({'scales':scales.tolist(),'mean_iou':value});return -value
    lo,hi=np.log(cfg['scale_bounds']);grid=np.linspace(lo,hi,cfg['grid_points']);best=min((objective(np.array([a,b])),a,b) for a in grid for b in grid);opt=minimize(objective,[best[1],best[2]],method='Powell',bounds=[(lo,hi),(lo,hi)],options={'xtol':1e-9,'ftol':1e-12,'maxiter':200})
    assert opt.success,opt.message
    # Retain the best of coarse grid, identity, and optimizer; all selection is train-only.
    objective(np.zeros(2));winner=max(trace,key=lambda x:x['mean_iou']);scales=winner['scales'];base=float(np.mean([paired_iou(a,b).mean() for a,b in pairs]));params=dict(width_scale=scales[0],height_scale=scales[1],fit_mean_iou_before=base,fit_mean_iou_after=winner['mean_iou'],fit_pair_count=sum(len(a) for a,b in pairs),fit_sequences=split['fit_sequences'],fitting='fixed IoU>=0.5 one-to-one training correspondences; sequence-equal mean IoU; logscale grid then bounded Powell; identity included',center_fixed=True,state_feedback=False)
    dump(R/'parameters.json',params);dump(R/'metrics.json',params);dump(R/'fit_sequences.json',stats);dump(R/'optimization_trace.json',trace)
    print(json.dumps(params),flush=True)

def eval_phase(B,E,R,cfg,split,phase,inputs):
    if phase=='dev':param=E/'results/run_001/parameters.json';domain='train';names=split['dev_sequences'];source=B/cfg['train_tracks']
    else:param=E/'results/run_002/frozen_parameters.json';domain='test';source=B/cfg['test_tracks'];names=sorted(p.stem for p in source.glob('UAVSwarm-*.txt'));assert len(names)==36
    inputs[str(param.relative_to(B))]=sha(param);params=json.loads(param.read_text());scales=[params['width_scale'],params['height_scale']];summary={};invariants={}
    # Produce predictions without loading evaluation GT. GT enters only evaluate() below.
    for arm in ['A','scaled']:
        target=R/arm/'tracks';target.mkdir(parents=True)
        for name in names:
            p=source/f'{name}.txt';inputs[str(p.relative_to(B))]=sha(p);raw=read(p);out=target/p.name
            if arm=='A':shutil.copyfile(p,out)
            else:
                changed=raw.copy();changed[:,2:6]=scale_boxes(raw[:,2:6],scales)
                text=[]
                for row,original_line in zip(changed,p.read_text().splitlines()):
                    fields=original_line.split(',');fields[2:6]=[f'{v:.10f}' for v in row[2:6]];text.append(','.join(fields)+'\n')
                out.write_text(''.join(text));saved=read(out);center_error=float(abs((saved[:,2:4]+saved[:,4:6]/2)-(raw[:,2:4]+raw[:,4:6]/2)).max());assert center_error<1e-8
                assert np.array_equal(saved[:,[0,1,6,7,8,9]],raw[:,[0,1,6,7,8,9]])
                assert len({tuple(r[:2]) for r in saved})==len(saved)
                invariants[name]={'rows':len(saved),'max_center_error_pixels':center_error,'frame_id_score_unchanged':True,'unique_frame_id':True}
        for name in names:
            p=B/cfg['dataset']/domain/name/'gt/gt.txt';inputs[str(p.relative_to(B))]=sha(p)
        metrics=evaluate(B/cfg['dataset'],domain,target,R/arm/'metrics.json',names);summary[arm]=metrics['overall'];print(phase,arm,json.dumps(metrics['overall']),flush=True)
    dump(R/'invariants.json',invariants);dump(R/'metrics.json',summary)
    if phase=='dev':
        shutil.copyfile(param,R/'frozen_parameters.json');assert sha(param)==sha(R/'frozen_parameters.json');dump(R/'freeze.json',{'parameters_identical_to_training':True,'sha256':sha(param),'dev_used_to_select_or_modify_parameters':False,'test_not_read_yet':True,'decision':'evaluate unchanged training-fit calibration on test regardless of dev outcome'})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--phase',choices=['fit','dev','test'],required=True);ap.add_argument('--run',required=True);args=ap.parse_args();B=args.root;E=B/'experiments/exp159_box_scale_calibration';W=Path(__file__).resolve().parent
    git=lambda *x:subprocess.check_output(['git','-C',str(W),*x],text=True).strip()
    assert not git('status','--porcelain')
    cfg=json.loads((E/'configs/protocol.json').read_text());split=json.loads((E/'configs/split_source.json').read_text());assert not set(split['fit_sequences'])&set(split['dev_sequences']);R=E/'results'/args.run;R.mkdir(parents=True,exist_ok=False)
    inputs={str(p.relative_to(B)):sha(p) for p in [E/'configs/protocol.json',E/'configs/split_source.json',W/'scale_calibration.py',W/'homa/evaluate.py',W/'tools/evaluate_uavswarm_mot.py']}
    # Verify frozen detector cache identity even though this output-only control reuses A trajectories.
    domains=['test'] if args.phase=='test' else ['train']
    for domain in domains:
        cache=B/cfg[domain+'_cache'];meta=json.loads((cache/'export_summary.json').read_text());inputs[str((cache/'export_summary.json').relative_to(B))]=sha(cache/'export_summary.json')
        names=split['fit_sequences'] if args.phase=='fit' else split['dev_sequences'] if args.phase=='dev' else [x['sequence'] for x in meta['sequences']]
        for name in names:
            p=cache/name/'det.txt';actual=sha(p);assert actual==next(x['det_sha256'] for x in meta['sequences'] if x['sequence']==name);inputs[str(p.relative_to(B))]=actual
    manifest={'phase':args.phase,'status':'running','private_repository':'git@github.com:YUSIO/ByteTrack-MOT.git','branch':'exp/159-box-scale','commit':git('rev-parse','HEAD'),'base_commit':'a7efabe57c7ddd72e51ef92bbf29a005f084137f','clean':True,'python_executable':sys.executable,'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'host':platform.node(),'device':'CPU: two-parameter numerical fit and evaluator; no neural inference needed','seed':None,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol':cfg,'canonical_run':True};dump(R/'manifest.yaml',manifest);(R/'command.txt').write_text(' '.join([sys.executable,*sys.argv])+'\n');rc=0
    with (R/'stdout.log').open('w') as stdout,(R/'stderr.log').open('w') as stderr,contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
        try:
            if args.phase=='fit':fit(B,E,R,cfg,split,inputs)
            else:eval_phase(B,E,R,cfg,split,args.phase,inputs)
        except Exception:traceback.print_exc();rc=1
    dump(R/'inputs.json',inputs);manifest.update(status='completed' if rc==0 else 'failed',exit_code=rc,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());dump(R/'manifest.yaml',manifest);(R/'exit_code.txt').write_text(str(rc)+'\n');(R/'report.md').write_text('# 运行记录\n\n'+('完成，详细解释见实验报告。' if rc==0 else '失败，证据保留。')+'\n');(R/'payload_checksums.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(R))+'\n' for p in sorted(R.rglob('*')) if p.is_file() and p.name!='payload_checksums.sha256'));print(args.phase,'exit',rc,flush=True);sys.exit(rc)
if __name__=='__main__':main()
