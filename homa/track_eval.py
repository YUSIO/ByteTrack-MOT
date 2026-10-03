import json
import subprocess
import sys
from pathlib import Path
from .evaluate import evaluate


def main():
    args=sys.argv[1:]
    output=Path(args[args.index('--output')+1]); output.mkdir(parents=True,exist_ok=False)
    track_args=args.copy(); track_args[track_args.index('--output')+1]=str(output/'tracks')
    subprocess.run([sys.executable,'-m','homa.track',*track_args],check=True)
    summary=json.loads((output/'tracks/summary.json').read_text())
    result=evaluate(Path(summary['args']['data']),summary['args']['split'],output/'tracks',output/'metrics.json',[s['sequence'] for s in summary['sequences']])
    print(json.dumps(result['overall']),flush=True)

if __name__=='__main__':main()
