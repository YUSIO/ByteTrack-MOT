"""Exp190: registered regional features; adapted from Exp180 fixed OOF probe."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,csv,hashlib,json,os,platform,shlex,subprocess,sys,time,traceback
SOURCE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(SOURCE))
import cv2,numpy as np,torch,sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score
from fctrack.verifier import load,seed_crop,scalars
from fctrack.weak_birth_gate import features
from fctrack.spatial_readout import region_weights, regional_means

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def git(*a):return subprocess.check_output(['git','-C',str(SOURCE),*a],text=True).strip()

def weights(rows):
    key=lambda r:(r['sequence'],r['frame'],r['owner'])
    count=Counter(key(r) for r in rows if int(r['label']))
    w=np.array([1/count[key(r)] if int(r['label']) else 1 for r in rows],float)
    seq=np.array([r['sequence'] for r in rows]);y=np.array([int(r['label']) for r in rows])
    for s in sorted(set(seq)):w[seq==s]/=w[seq==s].sum()
    for label in [0,1]:w[y==label]/=w[y==label].sum()
    return w*len(w)/w.sum()

def score_metrics(y,p,rows,cfg):
    accepted=p>=cfg['threshold'];positive=accepted&(y==1)
    gt={(r['sequence'],r['frame'],r['owner']) for i,r in enumerate(rows) if positive[i]}
    scenes={cfg['scene_for_sequence'][r['sequence']] for i,r in enumerate(rows) if positive[i]}
    return dict(AP=float(average_precision_score(y,p)),AUC=float(roc_auc_score(y,p)),
                accepted=int(accepted.sum()),true_positive=int(positive.sum()),
                false_positive=int((accepted&(y==0)).sum()),precision=float(positive.sum()/accepted.sum()) if accepted.any() else 0.,
                positive_unique_gt_frames=len(gt),positive_scenes=len(scenes),scene_keys=sorted(scenes))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--run',default='run_001');ap.add_argument('--check-only',action='store_true');a=ap.parse_args()
    root=a.root.resolve();e=root/'experiments/exp190_weak_birth_spatial_readout'
    cfg=read(e/'configs/probe.json');source=read(e/cfg['source_record']);inputs=read(e/cfg['input_record'])
    assert not git('status','--porcelain') and git('rev-parse','HEAD')==source['commit']
    for p,h in inputs.items():assert sha(root/p)==h,p
    rows=list(csv.DictReader((root/cfg['sample']).open()));assert {r['group'] for r in rows}=={'fit'}
    scene=np.array([cfg['scene_for_sequence'][r['sequence']] for r in rows]);fold=np.array([cfg['fold_for_scene'][s] for s in scene])
    y=np.array([int(r['label']) for r in rows]);assert len(rows)==2902 and y.sum()==69
    fold_info={}
    for f in range(3):
        train=fold!=f;test=~train
        assert set(scene[train]).isdisjoint(scene[test]) and len(set(y[train]))==2 and len(set(y[test]))==2
        fold_info[str(f)]=dict(train_scenes=sorted(set(scene[train])),heldout_scenes=sorted(set(scene[test])),
                              train_positive=int(y[train].sum()),heldout_positive=int(y[test].sum()))
    if a.check_only:
        print(json.dumps(dict(rows=len(rows),folds=fold_info,input_files=len(inputs),source_clean=True),indent=2));return 0
    torch.set_num_threads(1);cv2.setNumThreads(1);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    r=e/'results'/a.run;r.mkdir(parents=True,exist_ok=False)
    manifest=dict(run=a.run,phase='frozen_feature_oof_probe',status='running',source=source,dirty=False,device='cpu',threads=1,
                  seed=cfg['seed'],python_executable=sys.executable,python=sys.version,host=platform.node(),
                  versions={'numpy':np.__version__,'torch':torch.__version__,'sklearn':sklearn.__version__,'opencv':cv2.__version__},
                  input_manifest_sha256=sha(e/cfg['input_record']),config_sha256=sha(e/'configs/probe.json'),script_sha256=sha(Path(__file__)),
                  cal_read=False,dev_read=False,test_read=False,neural_weights_updated=False,
                  start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    save(r/'manifest.yaml',manifest);save(r/'frozen_config.json',cfg);save(r/'folds.json',fold_info);save(r/'input_sha256.json',inputs)
    (r/'command.txt').write_text(shlex.join([sys.executable,*sys.argv])+'\n')
    out=open(r/'stdout.log','w',buffering=1);err=open(r/'stderr.log','w',buffering=1);os.dup2(out.fileno(),1);os.dup2(err.fileno(),2);rc=1
    try:
        model=load('seed',root/cfg['seed_weights'],torch.device('cpu'));model.requires_grad_(False)
        col=lambda k:np.array([float(z[k]) for z in rows])
        raw=np.column_stack([col(k) for k in ['x','y','w','h']]);reg=np.column_stack([col(k) for k in ['rx','ry','rw','rh']])
        logged_p=col('p')
        geom=features(col('score'),logged_p,raw,reg,col('proposal_cover'),col('regressed_cover'))
        spatial_map=np.zeros((len(rows),256,4,4),np.float32)
        visual=np.zeros((len(rows),512),np.float32);parity={'p_max_error':0.,'box_max_error':0.};groups=defaultdict(list)
        for i,z in enumerate(rows):groups[(z['sequence'],int(z['frame']))].append(i)
        with torch.inference_mode():
            for (seq,frame),idx in groups.items():
                im=cv2.imread(str(root/cfg['dataset_root']/'train'/seq/'img1'/f'{frame:06d}.jpg'));assert im is not None
                for start in range(0,len(idx),cfg['batch_size']):
                    ids=idx[start:start+cfg['batch_size']];b=raw[ids];side=np.sqrt(b[:,2]*b[:,3]);centre=b[:,:2]+b[:,2:]/2
                    crops=torch.from_numpy(np.stack([seed_crop(im,*cc,ss) for cc,ss in zip(centre,side)]))
                    h=model.body(crops);v=torch.cat((h.mean((2,3)),h.amax((2,3))),1)
                    output=model.head(torch.cat((v,torch.from_numpy(scalars(side,np.full(len(ids),im.shape[1],float)))),1)).numpy().astype(float)
                    predp=1/(1+np.exp(-output[:,0]));wh=b[:,2:]*np.exp(output[:,3:5]);nc=centre+output[:,1:3]*side[:,None]
                    predreg=np.column_stack((nc-wh/2,wh))
                    pe=float(np.max(np.abs(predp-logged_p[ids])))
                    be=float(np.max(np.abs(predreg-reg[ids])))
                    parity['p_max_error']=max(parity['p_max_error'],pe);parity['box_max_error']=max(parity['box_max_error'],be)
                    assert pe<=cfg['parity']['p_atol'] and be<=cfg['parity']['box_atol_pixels'],(seq,frame,pe,be)
                    visual[ids]=v.numpy();spatial_map[ids]=h.numpy()
        assert np.isfinite(visual).all();save(r/'parity.json',parity)
        roi_weights=region_weights(raw,reg)
        regional=regional_means(spatial_map,roi_weights)
        regional_roll=regional_means(spatial_map,roi_weights,roll=True)
        variants=dict(GV=visual,GS=np.column_stack((visual,regional)),GS_roll=np.column_stack((visual,regional_roll)))
        np.savez_compressed(r/'features.npz',geometry=geom,visual=visual,spatial_map=spatial_map,
                            roi_weights=roi_weights,regional=regional,regional_roll=regional_roll,labels=y,fold=fold)
        predictions={};result={};(r/'fold_models').mkdir()
        for name in cfg['models']:
            pred=np.full(len(rows),np.nan)
            for f in range(3):
                tr=fold!=f;te=~tr;gs=StandardScaler().fit(geom[tr]);tx=gs.transform(geom[tr]);vx=gs.transform(geom[te])
                parameters={'geometry_mean':gs.mean_,'geometry_scale':gs.scale_}
                if name!='G':
                    values=variants[name]
                    vs=StandardScaler().fit(values[tr]);pca=PCA(n_components=cfg['pca_components'],whiten=cfg['pca_whiten'],svd_solver='full')
                    tv=pca.fit_transform(vs.transform(values[tr]));vv=pca.transform(vs.transform(values[te]));tx=np.column_stack((tx,tv));vx=np.column_stack((vx,vv))
                    parameters.update(visual_mean=vs.mean_,visual_scale=vs.scale_,pca_mean=pca.mean_,pca_components=pca.components_,pca_variance=pca.explained_variance_)
                lr=LogisticRegression(C=cfg['C'],max_iter=cfg['max_iter'],random_state=cfg['seed'])
                lr.fit(tx,y[tr],sample_weight=weights([z for i,z in enumerate(rows) if tr[i]]));assert lr.n_iter_.max()<cfg['max_iter']
                pred[te]=lr.predict_proba(vx)[:,1];parameters.update(coef=lr.coef_,intercept=lr.intercept_)
                np.savez_compressed(r/'fold_models'/f'{name}_{f}.npz',**parameters)
            assert np.isfinite(pred).all();predictions[name]=pred;result[name]=score_metrics(y,pred,rows,cfg)
            result[name]['per_fold']={str(f):score_metrics(y[fold==f],pred[fold==f],[z for i,z in enumerate(rows) if fold[i]==f],cfg) for f in range(3)}
        reference=list(csv.DictReader((root/cfg['reference_predictions']).open()))
        assert len(reference)==len(rows)
        for x,z in zip(reference,rows):assert all(x[k]==z[k] for k in ['sequence','frame','index','label'])
        parity['reference_prediction_max_error']={k:float(np.max(np.abs(predictions[k]-np.array([float(z[k]) for z in reference])))) for k in ['G','GV']}
        assert max(parity['reference_prediction_max_error'].values())<=cfg['reference_atol'],parity
        g=cfg['gate'];v=result['GS']
        gates={f'ap_vs_{name}':v['AP']-result[name]['AP']>=g['ap_delta'] for name in ['G','GV','GS_roll']}
        gates.update(precision=v['precision']>=g['precision'],positive_gt_frames=v['positive_unique_gt_frames']>=g['positive_unique_gt_frames'],
                     positive_scenes=v['positive_scenes']>=g['positive_scenes'])
        save(r/'metrics.json',dict(models=result,gates=gates,passed=all(gates.values()),parity=parity))
        with (r/'oof_predictions.csv').open('w') as file:
            writer=csv.DictWriter(file,fieldnames=['sequence','frame','index','owner','label','scene','fold',*cfg['models']]);writer.writeheader()
            for i,z in enumerate(rows):writer.writerow({**{k:z[k] for k in ['sequence','frame','index','owner','label']},'scene':scene[i],'fold':int(fold[i]),**{k:float(v[i]) for k,v in predictions.items()}})
        for p,h in inputs.items():assert sha(root/p)==h,p
        assert not git('status','--porcelain') and git('rev-parse','HEAD')==source['commit']
        manifest['status']='completed';rc=0;print(json.dumps(dict(metrics=result,gates=gates),indent=2),flush=True)
    except BaseException:traceback.print_exc();manifest['status']='failed'
    finally:
        manifest['end_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save(r/'manifest.yaml',manifest)
        (r/'exit_code.txt').write_text(str(rc)+'\n');sys.stdout.flush();sys.stderr.flush()
        save(r/'output_sha256.json',{str(p.relative_to(r)):sha(p) for p in sorted(r.rglob('*')) if p.is_file() and p.name!='output_sha256.json'})
    return rc

def read(p):return json.loads(p.read_text())
if __name__=='__main__':raise SystemExit(main())
