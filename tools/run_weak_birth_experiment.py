"""Immutable train-only Exp179 execution and strict evaluator aggregation."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
import platform
import random
import shlex
import subprocess
import sys
import time
import traceback

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def save(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def git(*args): return subprocess.check_output(['git','-C',str(SOURCE),*args],text=True).strip()


def evaluate(root,cfg,tracks):
    import numpy as np
    from trackeval.metrics import HOTA,CLEAR,Identity
    source=root/cfg['evaluator']
    spec=importlib.util.spec_from_file_location('pinned_eval',source)
    ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    ev.mm.lap.default_solver='lap'
    metrics={'hota':HOTA(),'clear':CLEAR({'THRESHOLD':.5,'PRINT_CONFIG':False}),
             'identity':Identity({'THRESHOLD':.5,'PRINT_CONFIG':False})}
    raw={k:{} for k in metrics};acc={}
    for name in cfg['cell']['sequences']:
        g=ev.read_mot_boxes(root/cfg['cell']['dataset_root']/'train'/name/'gt/gt.txt',1)
        p=ev.read_mot_boxes(tracks/f'{name}.txt',-1)
        d=ev.trackeval_data(g,p);d['num_timesteps']=len(d['gt_ids'])
        for k,m in metrics.items():raw[k][name]=m.eval_sequence(d)
        acc[name]=ev.strict_mot_accumulator(g,p)
    def compact(h,c,i,mot):
        return dict(HOTA=float(np.mean(h['HOTA'])*100),DetA=float(np.mean(h['DetA'])*100),
                    AssA=float(np.mean(h['AssA'])*100),MOTA=float(c['MOTA']*100),IDF1=float(i['IDF1']*100),
                    FP=int(c['CLR_FP']),FN=int(c['CLR_FN']),IDSW=int(c['IDSW']),Frag=int(c['Frag']),
                    GT=int(c['CLR_TP']+c['CLR_FN']),motmetrics=ev.serialise_mot_row(mot))
    groups={};per={}
    for group,names in cfg['groups'].items():
        mt=ev.mm.metrics.create().compute_many([acc[n] for n in names],names=names,metrics=ev.MOT_FIELDS,generate_overall=True)
        combined={k:m.combine_sequences({n:raw[k][n] for n in names}) for k,m in metrics.items()}
        groups[group]=compact(combined['hota'],combined['clear'],combined['identity'],mt.loc['OVERALL'])
        for n in names:per[n]=compact(raw['hota'][n],raw['clear'][n],raw['identity'][n],mt.loc[n])
    return dict(groups=groups,per_sequence=per,evaluator_sha256=sha(source),
                frame_policy='strict_union',clear_identity_iou=.5,hota_iou=[.05,.95])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--experiment',default='experiments/exp179_weak_birth_train')
    ap.add_argument('--config',default='configs/protocol.json')
    ap.add_argument('--run',required=True);a=ap.parse_args();root=a.root.resolve();e=root/a.experiment
    config=e/a.config
    cfg=json.loads(config.read_text());cell=cfg['cell'];rs=cfg['runs'][a.run]
    assert cell['split']=='train' and all(int(n[-2:])%2 for n in cell['sequences'])
    assert sorted(sum([cfg['groups'][k] for k in ['fit','calibration','dev']],[]))==sorted(cell['sequences'])
    assert cfg['device']=='cpu' and not git('status','--porcelain')
    source_record=e/cfg.get('source_record','preflight/source.json')
    input_record=e/cfg.get('input_record','preflight/inputs.json')
    expected=json.loads(source_record.read_text());assert git('rev-parse','HEAD')==expected['commit']
    inputs=json.loads(input_record.read_text())
    for p,h in inputs.items(): assert sha(root/p)==h,p
    import cv2,numpy as np,torch
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    torch.set_num_threads(1);cv2.setNumThreads(1)
    from fctrack.run import run_arm
    from fctrack.verifier import load
    load('seed',root/cell['verifiers']['seed']['weights'],torch.device('cpu')).eval()
    r=e/'results'/a.run;r.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',phase='track_eval',run=a.run,name=rs['name'],source=expected,
                  private_repository=cfg['private_repository'],base_commit=cfg['base_commit'],dirty=False,
                  config_path=str(config.relative_to(root)),config_sha256=sha(config),input_manifest_sha256=sha(input_record),
                  input_files=len(inputs),script_sha256=sha(Path(__file__)),device='cpu',threads=1,
                  seed=cfg['seed'],host=platform.node(),python_executable=sys.executable,python=sys.version,
                  versions={k:importlib.metadata.version(k) for k in ['torch','numpy','scipy','trackeval','motmetrics','lap']},
                  start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),test_read=False,
                  detector_seen_train=True,dataset=cell,groups=cfg['groups'],oracle=bool(rs['weak_birth'] and rs['weak_birth']['rule']=='oracle'))
    save(r/'manifest.yaml',manifest);save(r/'frozen_config.json',cfg)
    (r/'command.txt').write_text(shlex.join([sys.executable,*sys.argv])+'\n')
    stdout=open(r/'stdout.log','w',buffering=1);stderr=open(r/'stderr.log','w',buffering=1)
    os.dup2(stdout.fileno(),1);os.dup2(stderr.fileno(),2);rc=1
    try:
        arm=dict(cfg['arm'])
        if rs['weak_birth']:arm['weak_birth']=rs['weak_birth']
        if rs.get('weak_observe_sequences') is not None:arm['weak_observe_sequences']=rs['weak_observe_sequences']
        names,stats,digests=run_arm(root,cfg,cell,arm,r/'tracks','cpu',0)
        assert names==cell['sequences']
        save(r/'stats.json',stats);save(r/'tracks_sha256.json',digests)
        save(r/'metrics.json',evaluate(root,cfg,r/'tracks'))
        for p,h in inputs.items():assert sha(root/p)==h,p
        assert git('rev-parse','HEAD')==expected['commit'] and not git('status','--porcelain')
        manifest['status']='completed';rc=0
    except BaseException:traceback.print_exc();manifest['status']='failed'
    finally:
        manifest['end_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save(r/'manifest.yaml',manifest)
        (r/'exit_code.txt').write_text(str(rc)+'\n');sys.stdout.flush();sys.stderr.flush()
        save(r/'output_sha256.json',{str(p.relative_to(r)):sha(p) for p in sorted(r.rglob('*')) if p.is_file() and p.name!='output_sha256.json'})
    return rc


if __name__=='__main__':raise SystemExit(main())
