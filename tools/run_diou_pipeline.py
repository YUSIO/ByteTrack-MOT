"""Immutable train -> frozen cache -> detection evaluation execution for Exp199."""
import datetime, hashlib, json, os, platform, shlex, subprocess, sys
from pathlib import Path
import torch, torchvision, numpy, cv2
BASE=Path('/home/yusio/projects/MOT')
ROOT=BASE/'experiments/exp199_yolox_x_diou'
CODE=BASE/'worktrees/bytetrack-exp199-diou'
DATA=BASE/'datasets/UAVSwarm-dataset-master'
PY=BASE/'.venv/bin/python'
EXP='exps/example/uavswarm/yolox_x_uavswarmv1_diou.py'
SHA=json.loads((ROOT/'configs/protocol_v1.json').read_text())['code_commit']
WEIGHTS=BASE/'experiments/exp141_bytetrack_yolox_x_uavswarmv1/pretrained/yolox_x.pth'
GPU='0'

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d): Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def git(*args):return subprocess.check_output(['git','-C',str(CODE),*args],text=True).strip()
def manifest(run,m):write(run/'manifest.yaml',m) # JSON is valid YAML.
def phase(name,kind,cmd,parent=None,extra=None):
 run=ROOT/'results'/name;run.mkdir(parents=True,exist_ok=False)
 assert git('rev-parse','HEAD')==SHA and not git('status','--porcelain')
 m={'run_id':name,'phase':kind,'status':'running','canonical_run':name,'parent_run':parent,'started_utc':now(),'private_repository':'git@github.com:YUSIO/ByteTrack-MOT.git','branch':'exp/199-yolox-x-diou','code_commit':SHA,'code_dirty':False,'base_commit':'6d0b3b7747d3d59cab39cc7e697da481e1885276','upstream_commit':'d1bf0191adff59bc8fcfeaa0b33d3d1642552a99','python_executable':str(PY),'python_version':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':numpy.__version__,'opencv':cv2.__version__,'cuda':torch.version.cuda,'host':platform.node(),'physical_gpu_index':0,'visible_gpu_index':0,'seed':0,'dataset':'UAVSwarm V1','dataset_root':str(DATA),'config_path':EXP,'config_sha256':digest(CODE/EXP),'command':cmd,'environment':{k:os.environ[k] for k in ('CUDA_VISIBLE_DEVICES','PYTHONPATH','UAVSWARM_V1_ROOT')},'input_hashes':{str(p):digest(p) for p in [ROOT/'configs/protocol_v1.json',CODE/EXP,CODE/'tools/run_diou_pipeline.py',CODE/'tools/preflight_diou.py',BASE/'experiments/exp141_bytetrack_yolox_x_uavswarmv1/code/evaluate_cache.py']},'artifact_recovery':'pending','visualization':'waived; not requested',**(extra or {})}
 manifest(run,m)
 (run/'command.txt').write_text('cd '+shlex.quote(str(CODE))+'\n'+shlex.join(['env',*[k+'='+v for k,v in m['environment'].items()],*cmd])+'\n')
 (run/'gpu.txt').write_text(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=index,uuid,name,memory.total,driver_version','--format=csv,noheader'],text=True))
 with (run/'stdout.log').open('w') as stdout,(run/'stderr.log').open('w') as stderr:
  rc=subprocess.run(cmd,cwd=CODE,stdout=stdout,stderr=stderr).returncode
 (run/'exit_code.txt').write_text(str(rc)+'\n');m.update(status='completed' if rc==0 else 'failed',finished_utc=now(),exit_code=rc);manifest(run,m)
 (run/'report.md').write_text('# 执行状态\n\n'+('远端阶段正常退出，输出完整性仍待核验和本地回收。' if rc==0 else '执行失败；保留本次命令、日志和输出。禁止覆盖重试。')+'\n')
 if rc:raise RuntimeError(f'{name} exited {rc}')
 return run,m

def hash_outputs(run):
 paths=sorted(p for p in run.rglob('*') if p.is_file() and p.name not in ('output_checksums.sha256','manifest.yaml','report.md'))
 (run/'output_checksums.sha256').write_text(''.join(f'{digest(p)}  {p.relative_to(run)}\n' for p in paths))

def main():
 batch=int(sys.argv[1]);protocol=json.loads((ROOT/'configs/protocol_v1.json').read_text())
 assert protocol['batch']==batch and protocol['code_commit']==SHA
 assert digest(WEIGHTS)==protocol['pretrained_sha256']
 assert digest(DATA/'annotations/train.json')==protocol['train_json_sha256']
 assert json.loads((ROOT/'preflight'/f'batch{batch}_result.json').read_text())['status']=='passed'
 assert not (ROOT/'results/run_001').exists()
 processes=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 assert not processes, f'GPU{GPU} occupied: {processes}'
 os.environ.update(CUDA_VISIBLE_DEVICES=GPU,PYTHONPATH=str(CODE),UAVSWARM_V1_ROOT=str(DATA))
 train=ROOT/'results/run_001'
 cmd=[str(PY),'tools/train.py','-f',EXP,'-d','1','-b',str(batch),'--fp16','-c',str(WEIGHTS),'-expn','run_001','output_dir',str(train/'model')]
 run,m=phase('run_001','train',cmd,extra={'regression_loss':'1-IoU+center_sq/enclosing_diagonal_sq','divide_by_alpha':False,'regression_weight':5.0,'batch':batch,'base_lr':.001/64*batch,'epochs':80,'selection':'fixed_final_epoch_80_EMA','validation_split':None,'evaluation_during_training':'train_set_monitoring_only','pretrained_sha256':digest(WEIGHTS),'train_json_sha256':digest(DATA/'annotations/train.json'),'train_files_manifest_sha256':digest(ROOT/'preflight/train_files.sha256'),'train_images':6844,'train_boxes':49661})
 ckpt=run/'model/run_001/latest_ckpt.pth.tar'
 state=torch.load(ckpt,map_location='cpu',weights_only=True)
 assert state['start_epoch']==80,state['start_epoch']
 assert all(torch.isfinite(t).all() for t in state['model'].values())
 del state
 (run/'last.pt').symlink_to('model/run_001/latest_ckpt.pth.tar');(run/'selected.pt').symlink_to('last.pt')
 frozen=digest(ckpt);write(run/'selected_checkpoint.json',{'sha256':frozen,'epoch':80,'selection':'fixed_before_training','train_AP_best_used':False,'test_used_for_selection':False})
 m['selected_checkpoint_sha256']=frozen;manifest(run,m);hash_outputs(run)
 (run/'report.md').write_text('# 训练阶段完成\n\n80轮训练正常完成，固定选择第80轮EMA权重。`selected.pt`与`last.pt`一致；训练集AP最优权重不用于test。此阶段不宣称独立validation性能。测试与本地回收另行记录。\n')
 cache_run=ROOT/'results/run_002';cache=cache_run/'detections'
 cmd=[str(PY),'tools/export_uavswarm_detections.py','-f',EXP,'-c',str(run/'selected.pt'),'--data-root',str(DATA),'--split','test','--output',str(cache),'--conf','0.001','--nms','0.7','--use-exp-size','--batch','1','--device','cuda']
 cr,cm=phase('run_002','cache_export',cmd,'run_001',{'checkpoint_sha256':frozen,'test_size':[800,1440],'conf':.001,'nms':.7,'max_detections':'no_explicit_cap','batch':1,'precision':'fp32','tta':False,'gt_read':False})
 summary=json.loads((cache/'export_summary.json').read_text());assert len(summary['sequences'])==36 and summary['total_frames']==5754
 assert digest(ckpt)==frozen
 cm.update(sequences=36,frames=5754,detections=summary['total_detections']);manifest(cr,cm);hash_outputs(cr)
 gt=BASE/'experiments/exp088_yolox_s_1088_uavswarmv1_detector/data/uavswarmv1/test_gt'
 er=ROOT/'results/run_003'
 cmd=[str(PY),str(BASE/'experiments/exp141_bytetrack_yolox_x_uavswarmv1/code/evaluate_cache.py'),str(cache),str(gt),str(er/'metrics.json')]
 rr,rm=phase('run_003','test_eval',cmd,'run_002',{'checkpoint_sha256':frozen,'gt_root':str(gt),'evaluator':'pycocotools COCOeval bbox maxDets=1,10,100; GT clipped as Exp088','test_selection':'all_36_even_sequences'})
 rm['metrics']=json.loads((rr/'metrics.json').read_text());manifest(rr,rm);hash_outputs(rr)
 write(ROOT/'pipeline_status.json',{'status':'remote_completed_local_recovery_pending','finished_utc':now(),'runs':['run_001','run_002','run_003']})

if __name__=='__main__':
 try:main()
 except Exception as e:
  write(ROOT/'pipeline_status.json',{'status':'failed','time_utc':now(),'error':str(e)})
  raise
