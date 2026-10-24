"""Immutable two-arm frozen-cache development run; tracking and evaluation on CPU."""
import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import random
import shlex
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def compact(m):
    a=m['hota']; b=m['motmetrics']
    return {**{k:a[k] for k in ('HOTA','DetA','AssA','LocA')},
            **{k:b[v] for k,v in {'MOTA':'mota','IDF1':'idf1','IDSW':'num_switches',
              'FP':'num_false_positives','FN':'num_misses','Frag':'num_fragmentations',
              'GT':'num_objects','Pred':'num_predictions'}.items()}}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--cell',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); root=a.root.resolve(); code=Path(__file__).resolve().parents[1]
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.config.read_text()); c=cfg['cells'][a.cell]
    expected=json.loads(a.inputs.read_text())[a.cell]
    manifest={'phase':'track_eval','status':'running','started_utc':now(),
        'cell':a.cell,'dataset':c,'seed':cfg['seed'],'python_executable':sys.executable,
        'python_version':sys.version,'hardware':platform.uname()._asdict(),'device':'cpu',
        'cpu_count':os.cpu_count(),'load_at_start':os.getloadavg(),
        'private_repository':'git@github.com:YUSIO/ByteTrack-MOT.git',
        'branch':'exp/163-gmc-swarm-residual','upstream_repository':'https://github.com/FoundationVision/ByteTrack',
        'base_commit':'9b9b1611b0c17c408be78be90a9ab6167e91dc88',
        'config_path':str(a.config),'config_sha256':sha(a.config),
        'input_manifest_path':str(a.inputs),'input_manifest_sha256':sha(a.inputs),
        'execution_command':shlex.join([sys.executable,*sys.argv]),
        'evaluator':cfg['evaluator'],'output':str(out),'parent_run':None}
    dump(out/'manifest.yaml',manifest); dump(out/'frozen_config.json',cfg)
    dump(out/'input_sha256.json',expected)
    (out/'command.txt').write_text(manifest['execution_command']+'\n')
    (out/'stdout.log').touch(); (out/'stderr.log').touch()
    status=1
    try:
        manifest['commit']=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()
        manifest['dirty_state']=subprocess.check_output(['git','-C',str(code),'status','--porcelain'],text=True).strip()
        if manifest['dirty_state']:
            raise RuntimeError('formal code checkout is dirty')
        manifest['versions']={}
        for name in ('numpy','scipy','torch','motmetrics','trackeval','lap','cython_bbox'):
            m=importlib.import_module(name); manifest['versions'][name]=getattr(m,'__version__','unknown')
        random.seed(cfg['seed']); import numpy as np; np.random.seed(cfg['seed'])
        for rel,h in expected.items():
            if sha(root/rel)!=h:
                raise RuntimeError('input hash mismatch: '+rel)
        manifest['input_hashes_verified_before']=True
        dump(out/'manifest.yaml',manifest)
        env=os.environ.copy()
        env.update({'PYTHONPATH':str(code)+':'+str(code/'tools'),'CUDA_VISIBLE_DEVICES':'',
                    'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                    'PYTHONHASHSEED':str(cfg['seed'])})

        def run(cmd):
            with (out/'commands.txt').open('a') as f: f.write(shlex.join(cmd)+'\n')
            with (out/'stdout.log').open('a') as so,(out/'stderr.log').open('a') as se:
                subprocess.run(cmd,cwd=root,env=env,stdout=so,stderr=se,check=True)

        data={}; timings={}
        for arm,mode in cfg['arms'].items():
            start=time.monotonic()
            track=out/'tracks'/arm
            cmd=[sys.executable,str(code/'tools/track_uavswarm_detections.py'),
                '--dataset-root',str(root/c['dataset_root']),'--split','train',
                '--detections-root',str(root/c['detections']),'--gmc-dir',str(root/c['gmc']),
                '--sequences',*c['sequences'],'--output-dir',str(track),
                *cfg['tracker_args'],'--swarm-residual',mode]
            run(cmd); timings[arm+'_tracking_seconds']=time.monotonic()-start
            metric=out/(arm+'_metrics.json')
            run([sys.executable,str(code/'tools/evaluate_uavswarm_mot.py'),
                '--dataset-root',str(root/c['dataset_root']),'--split','train',
                '--sequences',*c['sequences'],'--tracker-results',str(track),
                '--output',str(metric),'--iou-threshold','0.5'])
            data[arm]=json.loads(metric.read_text())
            print(a.cell,arm,json.dumps(compact(data[arm]['overall'])),flush=True)
        overall={k:compact(v['overall']) for k,v in data.items()}
        delta={k:overall['kf_gmc_swarm'][k]-overall['kf_gmc'][k] for k in overall['kf_gmc']}
        per=[]
        for seq in c['sequences']:
            x=compact(data['kf_gmc']['per_sequence'][seq]); y=compact(data['kf_gmc_swarm']['per_sequence'][seq])
            per.append({'sequence':seq,**{k+'_base':v for k,v in x.items()},
                **{k+'_swarm':v for k,v in y.items()},**{'delta_'+k:y[k]-v for k,v in x.items()}})
        with (out/'per_sequence.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(per[0])); w.writeheader(); w.writerows(per)
        mechanism={}
        for path in (out/'tracks/kf_gmc_swarm').glob('*.residual.json'):
            mechanism[path.name.removesuffix('.residual.json')]=json.loads(path.read_text())
        result={'cell':a.cell,'protocol':c,'overall':overall,'delta':delta,
                'per_sequence':per,'mechanism':mechanism,'timings':timings,
                'dev_gate':delta['HOTA']>=0.5 and delta['MOTA']>=0 and delta['IDF1']>=0}
        dump(out/'metrics.json',result)
        for rel,h in expected.items():
            if sha(root/rel)!=h: raise RuntimeError('input changed during run: '+rel)
        manifest['input_hashes_verified_after']=True
        manifest['status']='completed'; status=0
    except Exception:
        manifest['status']='failed'
        with (out/'stderr.log').open('a') as f: traceback.print_exc(file=f)
        traceback.print_exc()
    finally:
        manifest['finished_utc']=now(); manifest['exit_code']=status
        dump(out/'manifest.yaml',manifest)
        (out/'exit_code.txt').write_text(str(status)+'\n')
        hashes=[f'{sha(f)}  {f.relative_to(out)}' for f in sorted(out.rglob('*'))
                if f.is_file() and f.name!='output_sha256.txt']
        (out/'output_sha256.txt').write_text('\n'.join(hashes)+'\n')
    return status


if __name__=='__main__':
    sys.exit(main())
