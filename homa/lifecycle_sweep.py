"""Frozen diagnostic plan; each changed lifecycle is a separate immutable run."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    plan=json.loads(a.plan.read_text());table=[]
    for item in plan['runs']:
        config=a.plan.parent/item['config']
        cfg=json.loads(config.read_text())
        run=a.root/'results'/item['run']
        cmd=[sys.executable,'-m','homa.track_eval','--data',str(a.data),
             '--detections',str(a.root/'inputs/validation_detections'),'--config',str(config),
             '--output',str(run/'evaluation'),'--split','train','--arm',item['arm'],
             '--window','8','--motion-window',str(item['motion_window']),
             '--kappa',str(item['kappa']),'--match-thresh','0.9','--reduction','sum',
             '--sequences',*cfg['train']['validation_sequences']]
        if item['arm']=='homa':
            cmd += ['--checkpoint',str(a.root/'results/run_028/model/best.pt'),
                    '--feature-cache',str(a.root/'features/train')]
        subprocess.run([sys.executable,'-m','homa.run','--run',str(run),'--phase','validation',
                        '--config',str(config),'--parent',item['parent'],'--',*cmd],check=True)
        metrics=json.loads((run/'evaluation/metrics.json').read_text())
        row=item|{'overall':metrics['overall']};table.append(row)
        (a.output/'metrics.json').write_text(json.dumps(table,indent=2)+'\n')
        print(json.dumps({'run':item['run'],'label':item['label'],
                          'HOTA':metrics['overall']['hota']['HOTA'],
                          'IDF1':metrics['overall']['motmetrics']['idf1'],
                          'IDSW':metrics['overall']['motmetrics']['num_switches']}),flush=True)


if __name__=='__main__':main()
