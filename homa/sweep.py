import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from .evaluate import evaluate


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--data',type=Path,required=True); p.add_argument('--config',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--selection',type=Path); p.add_argument('--first-run',type=int,required=True); p.add_argument('--parent',required=True)
    a=p.parse_args(); cfg=json.loads(a.config.read_text()); a.output.mkdir(parents=True,exist_ok=False)
    test=a.selection is not None; split='test' if test else 'train'
    names=sorted(p.name for p in (a.data/'test').glob('UAVSwarm-*') if p.is_dir()) if test else cfg['train']['validation_sequences']
    det=a.root/'inputs'/('test_detections' if test else 'validation_detections')
    if test:
        frozen=json.loads(a.selection.read_text()); configs=list(frozen['selected'].values())
    else:
        configs=[]; grid=cfg['validation_grid']
        for arm in grid['arms']:
            windows=grid['motion_window'] if arm in ('m2da','homa') else [8]
            kappas=grid['kappa'] if arm in ('m2da','homa') else [.1]
            for w,k,threshold in itertools.product(windows,kappas,grid['match_thresh']):configs.append({'arm':arm,'window':8,'motion_window':w,'kappa':k,'match_thresh':threshold,'reduction':'sum'})
    table=[]
    for i,c in enumerate(configs):
        label=f'{i:03d}_{c["arm"]}_T{c["motion_window"]}_k{c["kappa"]}_m{c["match_thresh"]}'
        run=a.root/'results'/f'run_{a.first_run+i:03d}'
        out=run/'evaluation'
        cmd=[sys.executable,'-m','homa.track_eval','--data',str(a.data),'--detections',str(det),'--config',str(a.config),'--checkpoint',str(a.checkpoint),'--output',str(out),'--split',split,'--feature-cache',str(a.root/'features'/split),'--sequences',*names]
        for key,value in c.items():cmd.extend(['--'+key.replace('_','-'),str(value)])
        wrapped=[sys.executable,'-m','homa.run','--run',str(run),'--phase','track_eval' if test else 'validation','--config',str(a.config),'--parent',a.parent,'--',*cmd]
        subprocess.run(wrapped,check=True)
        result=json.loads((out/'metrics.json').read_text())
        row={'run':run.name,'label':label,'config':c,'overall':result['overall']}; table.append(row)
        (a.output/'metrics.json').write_text(json.dumps(table,indent=2)+'\n'); print(json.dumps(row),flush=True)
    if not test:
        def key(row):
            o=row['overall']; return (o['hota']['HOTA'],o['motmetrics']['idf1'],-o['motmetrics']['num_switches'])
        selected={arm:max([r for r in table if r['config']['arm']==arm],key=key)['config'] for arm in cfg['validation_grid']['arms']}
        (a.output/'selection.json').write_text(json.dumps({'selected':selected,'criterion':'HOTA, IDF1, -IDSW; first grid member resolves exact ties','dataset':'MOT-train 12 validation sequences','official_test_read':False},indent=2)+'\n')

if __name__=='__main__':main()
