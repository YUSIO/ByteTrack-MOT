"""Immutable execution records shared by Exp192 collection and training."""
from pathlib import Path
import hashlib,json,os,platform,shlex,subprocess,sys,time,traceback
SOURCE=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def git(*args):return subprocess.check_output(['git','-C',str(SOURCE),*args],text=True).strip()
class Run:
    def __init__(self,root,experiment,config,run,phase):
        self.root=Path(root).resolve();self.e=self.root/experiment;self.cfg=read(self.e/config)
        self.inputs=read(self.e/self.cfg['input_record']);self.source=read(self.e/self.cfg['source_record'])
        assert not git('status','--porcelain') and git('rev-parse','HEAD')==self.source['commit']
        for p,h in self.inputs.items():assert sha(self.root/p)==h,p
        self.r=self.e/'results'/run;self.r.mkdir(exist_ok=False)
        self.manifest=dict(run=run,phase=phase,status='running',source=self.source,dirty=False,config=config,config_sha256=sha(self.e/config),input_manifest_sha256=sha(self.e/self.cfg['input_record']),input_files=len(self.inputs),python_executable=sys.executable,python=sys.version,host=platform.node(),start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),test_read=False,dev_read=False)
        save(self.r/'manifest.yaml',self.manifest);save(self.r/'frozen_config.json',self.cfg);save(self.r/'input_sha256.json',self.inputs)
        (self.r/'command.txt').write_text(shlex.join([sys.executable,*sys.argv])+'\n')
    def __enter__(self):
        self.out=open(self.r/'stdout.log','w',buffering=1);self.err=open(self.r/'stderr.log','w',buffering=1)
        os.dup2(self.out.fileno(),1);os.dup2(self.err.fileno(),2);return self
    def __exit__(self,kind,value,tb):
        rc=1 if kind else 0
        if kind:traceback.print_exception(kind,value,tb)
        else:
            try:
                for p,h in self.inputs.items():assert sha(self.root/p)==h,p
                assert not git('status','--porcelain') and git('rev-parse','HEAD')==self.source['commit']
            except BaseException:
                rc=1;traceback.print_exc()
        self.manifest.update(status='completed' if rc==0 else 'failed',end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        save(self.r/'manifest.yaml',self.manifest);(self.r/'exit_code.txt').write_text(str(rc)+'\n');sys.stdout.flush();sys.stderr.flush()
        save(self.r/'output_sha256.json',{str(p.relative_to(self.r)):sha(p) for p in sorted(self.r.rglob('*')) if p.is_file() and p.name!='output_sha256.json'})
        if rc and kind is None:raise RuntimeError('run final integrity check failed')
        return False
