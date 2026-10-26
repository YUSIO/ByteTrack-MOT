"""Collect fit-only fixed-box quality supervision directly from server-local caches."""
from pathlib import Path
import argparse,csv,hashlib,json,os,platform,shlex,subprocess,sys,time,traceback
SOURCE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(SOURCE))
import cv2,numpy as np,torch
from fctrack.verifier import load,verify,seed_crop,scalars
from fctrack.tracker import iou_matrix

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def git(*args):return subprocess.check_output(['git','-C',str(SOURCE),*args],text=True).strip()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--run',default='run_001');a=ap.parse_args()
    root=a.root.resolve();e=root/'experiments/exp181_weak_birth_quality';cfg=read(e/'configs/collect.json');source=read(e/cfg['source_record']);inputs=read(e/cfg['input_record'])
    assert not git('status','--porcelain') and git('rev-parse','HEAD')==source['commit']
    for p,h in inputs.items():assert sha(root/p)==h,p
    old=list(csv.DictReader((root/cfg['parent_samples']).open()));assert {z['group'] for z in old}=={'fit'}
    residual={(z['sequence'],int(z['frame']),int(z['index'])):z for z in old};assert len(residual)==len(old)==2902
    torch.set_num_threads(1);cv2.setNumThreads(1);torch.manual_seed(cfg['seed'])
    model=load('seed',root/cfg['seed_weights'],torch.device('cpu'));model.requires_grad_(False)
    r=e/'results'/a.run;r.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',phase='quality_supervision_collection',source=source,dirty=False,run=a.run,
        python_executable=sys.executable,python=sys.version,host=platform.node(),device='cpu',threads=1,seed=cfg['seed'],
        versions={'torch':torch.__version__,'numpy':np.__version__,'opencv':cv2.__version__},config_sha256=sha(e/'configs/collect.json'),
        inputs_sha256=sha(e/cfg['input_record']),script_sha256=sha(Path(__file__)),cal_read=False,dev_read=False,test_read=False,
        neural_weights_updated=False,start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    save(r/'manifest.yaml',manifest);save(r/'frozen_config.json',cfg);save(r/'inputs.json',inputs)
    (r/'command.txt').write_text(shlex.join([sys.executable,*sys.argv])+'\n')
    out=open(r/'stdout.log','w',buffering=1);err=open(r/'stderr.log','w',buffering=1);os.dup2(out.fileno(),1);os.dup2(err.fileno(),2);rc=1
    try:
        records=[];crops=[];scale_inputs=[];seen=set();maxp=maxb=0.;invalid=0
        for seq in cfg['fit_sequences']:
            seqdir=root/cfg['dataset_root']/'train'/seq
            cache=np.loadtxt(root/cfg['detections']/seq/'det.txt',delimiter=',',ndmin=2)
            gt=np.loadtxt(seqdir/'gt/gt.txt',delimiter=',',ndmin=2);gt=gt[gt[:,6]>=1]
            for frame in sorted(set(cache[:,0].astype(int))):
                rr=cache[cache[:,0]==frame]
                det=np.column_stack((rr[:,2:4],rr[:,2:4]+rr[:,4:6],rr[:,6])).astype(np.float32)
                tlwh=np.column_stack((det[:,:2],det[:,2:4]-det[:,:2])).astype(float)
                sel=np.flatnonzero((det[:,4]>=cfg['lo'])&(det[:,4]<cfg['hi']))
                if not len(sel):continue
                image=cv2.imread(str(seqdir/'img1'/f'{frame:06d}.jpg'));assert image is not None
                gg=gt[gt[:,0]==frame]
                for start in range(0,len(sel),cfg['batch_size']):
                    ids=sel[start:start+cfg['batch_size']]
                    assert (tlwh[ids,2:]>0).all()
                    pp,boxes=verify('seed',model,image,tlwh[ids],torch.device('cpu'))
                    for j,k in enumerate(ids):
                        key=(seq,frame,int(k));parent=residual.get(key)
                        if parent:
                            assert key not in seen;seen.add(key)
                            np.testing.assert_allclose(tlwh[k],[float(parent[x]) for x in ['x','y','w','h']],rtol=0,atol=1e-7)
                            preg=np.array([float(parent[x]) for x in ['rx','ry','rw','rh']]);pold=float(parent['p'])
                            pe=abs(pp[j]-pold);be=float(np.max(np.abs(boxes[j]-preg)));maxp=max(maxp,float(pe));maxb=max(maxb,be)
                            assert pe<=cfg['parity']['p_atol'] and be<=cfg['parity']['box_atol_pixels'],(key,pe,be)
                            pp[j]=pold;boxes[j]=preg
                    valid=(boxes[:,2]>1)&(boxes[:,3]>1);invalid+=int((~valid).sum())
                    if not valid.any():continue
                    ids=ids[valid];pp=pp[valid];boxes=boxes[valid]
                    p_reg,_=verify('seed',model,image,boxes,torch.device('cpu'))
                    overlaps=iou_matrix(boxes,gg[:,2:6])
                    for j,k in enumerate(ids):
                        key=(seq,frame,int(k));parent=residual.get(key);box=boxes[j]
                        best=int(np.argmax(overlaps[j])) if len(gg) else None
                        quality=float(overlaps[j,best]) if best is not None else 0.;label=int(quality>=.5);owner=int(gg[best,1]) if label else -1
                        if parent:assert label==int(parent['label']),key
                        side=np.sqrt(box[2]*box[3]);centre=box[:2]+box[2:]/2
                        crop=seed_crop(image,*centre,side)
                        crops.append(crop);scale_inputs.append(scalars(np.array([side]),np.array([image.shape[1]],float))[0])
                        records.append(dict(sequence=seq,scene=cfg['scene_for_sequence'][seq],frame=frame,index=int(k),
                            score=float(det[k,4]),p=float(pp[j]),p_reg=float(p_reg[j]),rx=float(box[0]),ry=float(box[1]),rw=float(box[2]),rh=float(box[3]),
                            quality=quality,label=label,owner=owner,residual=int(parent is not None),fold=cfg['fold_for_scene'][cfg['scene_for_sequence'][seq]]))
            print(seq,'cumulative_samples',len(records),flush=True)
        assert seen==set(residual)
        pos=[z for z in records if z['label']];res=[z for z in records if z['residual']]
        unique={(z['sequence'],z['frame'],z['owner']) for z in pos};scenes=sorted({z['scene'] for z in pos})
        data_gate=cfg['data_gate'];gates=dict(unique_positive_gt_frames=len(unique)>=data_gate['unique_positive_gt_frames'],
            positive_scenes=len(scenes)>=data_gate['positive_scenes'],residual_rows=len(res)==data_gate['residual_rows'],
            residual_positive=sum(z['label'] for z in res)==data_gate['residual_positive'])
        metrics=dict(samples=len(records),positive=len(pos),negative=len(records)-len(pos),unique_positive_gt_frames=len(unique),
            positive_scenes=scenes,residual_rows=len(res),residual_positive=sum(z['label'] for z in res),invalid_regression_boxes=invalid,
            parity={'p_max_error':maxp,'box_max_error':maxb},gates=gates,passed=all(gates.values()),
            original_raw_p097=dict(accepted=sum(z['p']>=.97 for z in res),positive=sum(z['p']>=.97 and z['label'] for z in res)),
            original_reg_crop_p097=dict(accepted=sum(z['p_reg']>=.97 for z in res),positive=sum(z['p_reg']>=.97 and z['label'] for z in res)))
        with (r/'samples.csv').open('w') as file:
            writer=csv.DictWriter(file,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        np.savez_compressed(r/'crops.npz',images=np.stack(crops),scalars=np.stack(scale_inputs))
        save(r/'metrics.json',metrics)
        for p,h in inputs.items():assert sha(root/p)==h,p
        assert not git('status','--porcelain') and git('rev-parse','HEAD')==source['commit']
        manifest['status']='completed';rc=0;print(json.dumps(metrics,indent=2),flush=True)
    except BaseException:traceback.print_exc();manifest['status']='failed'
    finally:
        manifest['end_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save(r/'manifest.yaml',manifest)
        (r/'exit_code.txt').write_text(str(rc)+'\n');sys.stdout.flush();sys.stderr.flush()
        save(r/'output_sha256.json',{str(p.relative_to(r)):sha(p) for p in sorted(r.rglob('*')) if p.is_file() and p.name!='output_sha256.json'})
    return rc

if __name__=='__main__':raise SystemExit(main())
