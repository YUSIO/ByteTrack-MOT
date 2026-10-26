"""Fit two preregistered weak-birth gates on fit only; calibration labels are diagnostics."""
from pathlib import Path
from collections import Counter
import argparse,csv,hashlib,json,os,platform,shlex,subprocess,sys,time,traceback
SOURCE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(SOURCE))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
import sklearn
from fctrack.weak_birth_gate import features,predict,FEATURE_NAMES

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def git(*args):return subprocess.check_output(['git','-C',str(SOURCE),*args],text=True).strip()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--run',default='run_015');a=ap.parse_args()
    root=a.root.resolve();e=root/'experiments/exp179_weak_birth_train';sample=e/'analysis/s_gate_samples.csv';admission=e/'analysis/s_gate_admission.json'
    expected=json.loads((e/'preflight/source_stage7.json').read_text());assert git('rev-parse','HEAD')==expected['commit'] and not git('status','--porcelain')
    admitted=json.loads(admission.read_text());assert admitted['passed'] and sha(sample)==admitted['sample_sha256']
    rows=list(csv.DictReader(sample.open()));assert set(r['group'] for r in rows)=={'fit','calibration'}
    col=lambda k:np.array([float(r[k]) for r in rows])
    raw=np.column_stack([col(k) for k in ['x','y','w','h']]);reg=np.column_stack([col(k) for k in ['rx','ry','rw','rh']])
    x=features(col('score'),col('p'),raw,reg,col('proposal_cover'),col('regressed_cover'));y=col('label').astype(int)
    fit=np.array([r['group']=='fit' for r in rows]);cal=~fit
    assert y[fit].sum()>0 and y[cal].sum()>0
    r=e/'results'/a.run;r.mkdir(parents=True,exist_ok=False)
    manifest=dict(phase='gate_fit',status='running',run=a.run,source=expected,dirty=False,seed=179,device='cpu',host=platform.node(),
      python_executable=sys.executable,python=sys.version,versions=dict(numpy=np.__version__,sklearn=sklearn.__version__),
      sample_sha256=sha(sample),admission_sha256=sha(admission),plan_sha256=sha(e/'preflight/stage6_gate_plan.md'),script_sha256=sha(Path(__file__)),
      learned_on='fit19_only',calibration_use='post_fit_diagnostics_only',dev_read=False,test_read=False,
      start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    save(r/'manifest.yaml',manifest);(r/'command.txt').write_text(shlex.join([sys.executable,*sys.argv])+'\n')
    out=open(r/'stdout.log','w',buffering=1);err=open(r/'stderr.log','w',buffering=1);os.dup2(out.fileno(),1);os.dup2(err.fileno(),2);rc=1
    try:
        fr=[z for z in rows if z['group']=='fit'];fy=y[fit]
        key=lambda z:(z['sequence'],z['frame'],z['owner'])
        multiplicity=Counter(key(z) for z in fr if int(z['label']))
        w=np.array([1/multiplicity[key(z)] if int(z['label']) else 1 for z in fr],float)
        seq=np.array([z['sequence'] for z in fr])
        for s in set(seq):w[seq==s]/=w[seq==s].sum()
        for label in [0,1]:w[fy==label]/=w[fy==label].sum()
        w*=len(w)/w.sum()
        diagnostics={}
        for name,n in [('PS',2),('PSG',8)]:
            mean=x[fit,:n].mean(0);std=x[fit,:n].std(0);std=np.where(std<1e-12,1,std)
            learner=LogisticRegression(C=.1,max_iter=1000,random_state=179)
            learner.fit((x[fit,:n]-mean)/std,fy,sample_weight=w)
            assert int(learner.n_iter_.max())<1000,'optimizer did not converge'
            model=dict(schema_version=1,name=name,feature_names=list(FEATURE_NAMES[:n]),mean=mean.tolist(),std=std.tolist(),
              coef=learner.coef_[0].tolist(),intercept=float(learner.intercept_[0]),C=.1,max_iter=1000,
              fit_rows=int(fit.sum()),fit_positive=int(y[fit].sum()),training_sample_sha256=sha(sample),code_commit=expected['commit'],
              score_semantics='class-balanced logistic score; not calibrated probability')
            pred=predict(model,x);reference=learner.predict_proba((x[:,:n]-mean)/std)[:,1]
            np.testing.assert_allclose(pred,reference,rtol=1e-12,atol=1e-12)
            save(r/f'model_{name}.json',model)
            diagnostics[name]={}
            for group,mask in [('fit',fit),('calibration',cal)]:
                diagnostics[name][group]=dict(rows=int(mask.sum()),positives=int(y[mask].sum()),
                  auc=float(roc_auc_score(y[mask],pred[mask])),ap=float(average_precision_score(y[mask],pred[mask])),
                  thresholds={str(t):dict(accepted=int((pred[mask]>=t).sum()),positive=int(((pred[mask]>=t)&(y[mask]==1)).sum()),
                    negative=int(((pred[mask]>=t)&(y[mask]==0)).sum())) for t in [.5,.8,.95]})
            diagnostics[name]['export_predict_max_error']=float(np.max(np.abs(pred-reference)))
        save(r/'metrics.json',diagnostics);print(json.dumps(diagnostics,indent=2),flush=True)
        assert sha(sample)==manifest['sample_sha256'] and not git('status','--porcelain')
        manifest['status']='completed';rc=0
    except BaseException:traceback.print_exc();manifest['status']='failed'
    finally:
        manifest['end_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save(r/'manifest.yaml',manifest)
        (r/'exit_code.txt').write_text(str(rc)+'\n');sys.stdout.flush();sys.stderr.flush()
        save(r/'output_sha256.json',{str(p.relative_to(r)):sha(p) for p in sorted(r.rglob('*')) if p.is_file() and p.name!='output_sha256.json'})
    return rc

if __name__=='__main__':raise SystemExit(main())
