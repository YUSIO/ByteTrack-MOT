"""Immutable execution wrapper. Each invocation must use a new run directory."""
import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path


def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--phase',required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--parent',default='none');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    a.run.mkdir(parents=True,exist_ok=False)
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip()
    if dirty:raise RuntimeError('formal code working tree is dirty')
    meta={'schema_version':1,'phase':a.phase,'status':'running','started_utc':utc(),'parent_run':a.parent,'command':command,'repository':'git@github.com:YUSIO/ByteTrack-MOT.git','branch':'exp/042-homatracker-yolo11s','commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'dirty':dirty,'base_commit':'758f44efe2fca50188e1a697013db8d819ad306c','upstream_commit':'d1bf0191adff59bc8fcfeaa0b33d3d1642552a99','input_manifest_sha256':hashlib.sha256((a.config.parent/'manifest.json').read_bytes()).hexdigest(),'config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest(),'python_executable':sys.executable,'python':sys.version,'platform':platform.platform(),'host':platform.node(),'pid':os.getpid()}
    (a.run/'command.txt').write_text(shlex.join(command)+'\n')
    (a.run/'manifest.yaml').write_text(json.dumps(meta,indent=2)+'\n')
    (a.run/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    with (a.run/'stdout.log').open('w') as out,(a.run/'stderr.log').open('w') as err:r=subprocess.run(command,stdout=out,stderr=err)
    (a.run/'exit_code.txt').write_text(str(r.returncode)+'\n')
    meta.update(status='completed' if r.returncode==0 else 'failed',exit_code=r.returncode,finished_utc=utc())
    (a.run/'manifest.yaml').write_text(json.dumps(meta,indent=2)+'\n')
    (a.run/'report.md').write_text(f'# 执行状态\n\n阶段：`{a.phase}`。状态：`{meta["status"]}`。退出码：`{r.returncode}`。\n\n本文件仅记录进程状态；指标解释和本地恢复校验由实验报告补充。\n')
    files=sorted(x for x in a.run.rglob('*') if x.is_file())
    with (a.run/'checksums.sha256').open('w') as f:
        for x in files:
            with x.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            f.write(f'{digest}  {x.relative_to(a.run)}\n')
    raise SystemExit(r.returncode)

if __name__=='__main__':main()
