"""Fixed, fit-only regression consistency probe; no output box or weights change."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,hashlib,json,os,platform,shlex,subprocess,sys,time,traceback
SOURCE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(SOURCE))
import cv2,numpy as np,torch
from fctrack.verifier import load,verify

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def git(*a):return subprocess.check_output(['git','-C',str(SOURCE),*a],text=True).strip()

def proposals(raw,offsets):
    raw=np.asarray(raw,float)
    out=np.repeat(raw[:,None,:],len(offsets)+1,axis=1)
    out[:,1:,:2]+=np.sqrt(raw[:,2]*raw[:,3])[:,None,None]*np.asarray(offsets)[None,:,:]
    return out

def consistency(p,boxes):
    """Every auxiliary result must be valid and overlap the fixed original result."""
    valid=np.isfinite(boxes).all(-1)&(boxes[:,:,2]>1)&(boxes[:,:,3]>1)&np.isfinite(p)
    safe=np.where(valid[:,:,None],boxes,0.)
    anchor=safe[:,:1];aux=safe[:,1:]
    wh=np.maximum(0.,np.minimum(anchor[:,:,:2]+anchor[:,:,2:],aux[:,:,:2]+aux[:,:,2:])-np.maximum(anchor[:,:,:2],aux[:,:,:2]))
    inter=wh[:,:,0]*wh[:,:,1]
    union=anchor[:,:,2]*anchor[:,:,3]+aux[:,:,2]*aux[:,:,3]-inter
    iou=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
    iou=np.where(valid[:,:1]&valid[:,1:],iou,0.)
    return np.where(valid,p,0.).min(1),iou.min(1),iou

def counts(rows,accepted,cfg):
    y=np.array([int(z['label']) for z in rows]);positive=accepted&(y==1)
    gt={(z['sequence'],z['frame'],z['owner']) for i,z in enumerate(rows) if positive[i]}
    scenes=sorted({cfg['scene_for_sequence'][z['sequence']] for i,z in enumerate(rows) if positive[i]})
    return dict(accepted=int(accepted.sum()),true_positive=int(positive.sum()),false_positive=int((accepted&(y==0)).sum()),
                precision=float(positive.sum()/accepted.sum()) if accepted.any() else 0.,unique_positive_gt_frames=len(gt),
                positive_scenes=len(scenes),scene_keys=scenes)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--run',default='run_001');a=ap.parse_args()
    root=a.root.resolve();e=root/'experiments/exp182_weak_birth_jitter';cfg=read(e/'configs/probe.json');source=read(e/cfg['source_record']);inputs=read(e/cfg['input_record'])
    assert not git('status','--porcelain') and git('rev-parse','HEAD')==source['commit']
    for p,h in inputs.items():assert sha(root/p)==h,p
    rows=list(csv.DictReader((root/cfg['sample']).open()));assert len(rows)==2902 and {z['group'] for z in rows}=={'fit'}
    assert sum(int(z['label']) for z in rows)==69
    assert not any(cfg[k] for k in ['cal_read','dev_read','test_read'])
    torch.set_num_threads(cfg['threads']);cv2.setNumThreads(1);torch.manual_seed(cfg['seed'])
    model=load('seed',root/cfg['seed_weights'],torch.device('cpu'));model.requires_grad_(False)
    r=e/'results'/a.run;r.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',run=a.run,phase='fixed_jitter_fit_probe',source=source,dirty=False,
        python_executable=sys.executable,python=sys.version,host=platform.node(),device='cpu',threads=1,seed=cfg['seed'],
        versions=dict(torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__),
        config_sha256=sha(e/'configs/probe.json'),inputs_sha256=sha(e/cfg['input_record']),script_sha256=sha(Path(__file__)),
        cal_read=False,dev_read=False,test_read=False,training=False,tracker_intervention=False,
        start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    save(r/'manifest.yaml',manifest);save(r/'frozen_config.json',cfg);save(r/'inputs.json',inputs)
    (r/'command.txt').write_text(shlex.join([sys.executable,*sys.argv])+'\n')
    out=open(r/'stdout.log','w',buffering=1);err=open(r/'stderr.log','w',buffering=1);os.dup2(out.fileno(),1);os.dup2(err.fileno(),2);rc=1
    try:
        groups=defaultdict(list)
        for i,z in enumerate(rows):groups[(z['sequence'],int(z['frame']))].append(i)
        raw=np.array([[float(z[k]) for k in ['x','y','w','h']] for z in rows]);q=proposals(raw,cfg['offsets'])
        ps=np.full(q.shape[:2],np.nan);bs=np.full(q.shape,np.nan);done=0
        for (seq,frame),idx in groups.items():
            image=cv2.imread(str(root/cfg['dataset_root']/'train'/seq/'img1'/f'{frame:06d}.jpg'));assert image is not None
            flat=q[idx].reshape(-1,4);pv=[];bv=[]
            for start in range(0,len(flat),cfg['batch_size']):
                p,b=verify('seed',model,image,flat[start:start+cfg['batch_size']],torch.device('cpu'));pv.extend(p);bv.extend(b)
            ps[idx]=np.asarray(pv).reshape(len(idx),-1);bs[idx]=np.asarray(bv).reshape(len(idx),-1,4)
            done+=len(idx)
            if done%100< len(idx):print('queries',done,flush=True)
        originalp=np.array([float(z['p']) for z in rows]);originalb=np.array([[float(z[k]) for k in ['rx','ry','rw','rh']] for z in rows])
        pe=float(np.max(np.abs(ps[:,0]-originalp)));be=float(np.max(np.abs(bs[:,0]-originalb)))
        assert pe<=cfg['parity_atol'] and be<=cfg['box_atol'],(pe,be)
        ps[:,0]=originalp;bs[:,0]=originalb
        pmin,imin,ious=consistency(ps,bs);score=pmin*imin
        accept=dict(P=originalp>=cfg['p_min'],Pmin=pmin>=cfg['p_min'],J=(pmin>=cfg['p_min'])&(imin>=cfg['iou_min']))
        from sklearn.metrics import average_precision_score,roc_auc_score
        y=np.array([int(z['label']) for z in rows]);ap0=float(average_precision_score(y,originalp));apj=float(average_precision_score(y,score))
        metrics={name:counts(rows,mask,cfg) for name,mask in accept.items()}
        gate=cfg['gate'];m=metrics['J'];gates=dict(precision=m['precision']>=gate['precision'],
            unique_positive_gt_frames=m['unique_positive_gt_frames']>=gate['unique_positive_gt_frames'],
            positive_scenes=m['positive_scenes']>=gate['positive_scenes'],AP_gain=apj-ap0>=gate['AP_gain'])
        per_sequence={}
        for seq in sorted({z['sequence'] for z in rows}):
            sel=np.array([z['sequence']==seq for z in rows]);rr=[z for i,z in enumerate(rows) if sel[i]]
            per_sequence[seq]={name:counts(rr,mask[sel],cfg) for name,mask in accept.items()}
        result=dict(queries=len(rows),positive=int(y.sum()),network_queries=int(ps.size),rules=metrics,
            scores=dict(P_AP=ap0,J_AP=apj,AP_gain=apj-ap0,P_AUC=float(roc_auc_score(y,originalp)),J_AUC=float(roc_auc_score(y,score))),
            gates=gates,passed=all(gates.values()),parity=dict(p_max_error=pe,box_max_error=be),per_sequence=per_sequence)
        save(r/'metrics.json',result);np.savez_compressed(r/'probes.npz',proposals=q,p=ps,boxes=bs,ious=ious,pmin=pmin,imin=imin,score=score)
        with (r/'query_predictions.csv').open('w') as file:
            writer=csv.DictWriter(file,fieldnames=['sequence','frame','index','owner','label','p','pmin','imin','score','P','Pmin','J']);writer.writeheader()
            for i,z in enumerate(rows):writer.writerow({**{k:z[k] for k in ['sequence','frame','index','owner','label','p']},
                'pmin':float(pmin[i]),'imin':float(imin[i]),'score':float(score[i]),**{k:int(v[i]) for k,v in accept.items()}})
        for p,h in inputs.items():assert sha(root/p)==h,p
        assert not git('status','--porcelain') and git('rev-parse','HEAD')==source['commit']
        manifest['status']='completed';rc=0;print(json.dumps({k:v for k,v in result.items() if k!='per_sequence'},indent=2),flush=True)
    except BaseException:traceback.print_exc();manifest['status']='failed'
    finally:
        manifest['end_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save(r/'manifest.yaml',manifest)
        (r/'exit_code.txt').write_text(str(rc)+'\n');sys.stdout.flush();sys.stderr.flush()
        save(r/'output_sha256.json',{str(p.relative_to(r)):sha(p) for p in sorted(r.rglob('*')) if p.is_file() and p.name!='output_sha256.json'})
    return rc

if __name__=='__main__':raise SystemExit(main())
