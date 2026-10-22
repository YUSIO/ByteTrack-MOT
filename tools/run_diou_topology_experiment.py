"""Immutable remote cached-detection runs with complete execution evidence."""
import argparse, datetime, hashlib, importlib.metadata, json, os, platform, shlex, subprocess, sys
from pathlib import Path

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--config',type=Path,required=True); a=ap.parse_args()
    root=a.root.resolve(); repo=Path(__file__).resolve().parents[1]; exp=root/'experiments/exp143_diou_topology_fidelity'
    spec=json.loads(a.config.read_text()); dataset=root/spec['dataset']
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
    if dirty: raise RuntimeError('formal code must be clean')
    env=os.environ.copy(); env.update(PYTHONPATH=str(repo),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for run in spec['runs']:
        out=exp/'results'/run['id']; out.mkdir(parents=True,exist_ok=False)
        cache=root/spec['caches'][run['cache']]
        inputs={str(a.config.relative_to(root)):sha(a.config)}
        for seq in sorted((dataset/'test').glob('UAVSwarm-*')):
            for p in [seq/'seqinfo.ini',seq/'gt/gt.txt',cache/seq.name/'det.txt']:
                inputs[str(p.relative_to(root))]=sha(p)
        track=[sys.executable,str(repo/'tools/track_uavswarm_detections.py'),'--dataset-root',str(dataset),'--detections-root',str(cache),'--split','test','--output-dir',str(out/'track_results')]
        for k,v in spec['tracker'].items(): track += ['--'+k.replace('_','-'),str(v)]
        track+=['--first-metric',run['first_metric'],'--topology-version',run.get('topology_version','paper')]
        if run['topology']: track+=['--topology']
        evaluate=[sys.executable,str(repo/'tools/evaluate_uavswarm_mot.py'),'--dataset-root',str(dataset),'--split','test','--tracker-results',str(out/'track_results'),'--output',str(out/'metrics.json')]
        (out/'command.txt').write_text('PYTHONPATH='+shlex.quote(str(repo))+' '+shlex.join(track)+'\n'+'PYTHONPATH='+shlex.quote(str(repo))+' '+shlex.join(evaluate)+'\n')
        manifest={'phase':'track_eval','status':'running','started_utc':now(),'run':run,'private_repository':'git@github.com:YUSIO/ByteTrack-MOT.git','branch':'exp/143-diou-topology-fidelity','commit':commit,'base_commit':'5e0d0c68d5b1e67b107c2202db885b533769d9dd','code_dirty':False,'python_executable':sys.executable,'python_version':sys.version,'platform':platform.platform(),'hostname':platform.node(),'seed':'not applicable; deterministic','device':'CPU','threads':1,'dependencies':{n:importlib.metadata.version(n) for n in ['numpy','scipy','torch','lap','motmetrics','trackeval']},'config':spec,'input_sha256':inputs,'evaluator_sha256':sha(repo/'tools/evaluate_uavswarm_mot.py'),'evaluator':'motmetrics CLEAR/ID IoU0.5 + TrackEval HOTA, GT/results union','visualization':'derived after metric comparison','detector_not_rerun':True}
        mp=out/'manifest.yaml'; mp.write_text(json.dumps(manifest,indent=2)+'\n')
        code=0
        with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
            for cmd in [track,evaluate]:
                code=subprocess.run(cmd,cwd=repo,env=env,stdout=stdout,stderr=stderr).returncode
                if code: break
        (out/'exit_code.txt').write_text(str(code)+'\n')
        manifest.update(status='completed' if code==0 else 'failed',exit_code=code,finished_utc=now())
        mp.write_text(json.dumps(manifest,indent=2)+'\n')
        (out/'output_sha256.txt').write_text(''.join(sha(p)+'  '+str(p.relative_to(out))+'\n' for p in sorted(out.rglob('*')) if p.is_file() and p.name!='output_sha256.txt'))
        print(run['id'],manifest['status'],flush=True)
        if code: raise SystemExit(code)
if __name__=='__main__': main()
