"""Read-only association capture; GT is used only after baseline replay."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,gzip,hashlib,subprocess,datetime,traceback,platform,argparse,shlex
import numpy as np
from fctrack.run import seq_info
from fctrack.tracker import FCTrack,wrap_stracks
from yolox.tracker.byte_tracker import BYTETracker,STrack
from yolox.tracker.basetrack import BaseTrack
from homa.evaluate import evaluate
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def capture(ts,ds,cost,matches):
 return {'tracks':[{'id':int(t.track_id),'state':int(t.state),'last_frame':int(t.frame_id),'last_box':np.asarray(t._det if hasattr(t,'_det') else t._tlwh,float).tolist(),'pred':np.asarray(t.tlbr,float).tolist()} for t in ts], 'detections':[{'box':np.asarray(d.tlbr,float).tolist(),'score':float(d.score)} for d in ds], 'cost':cost.tolist(),'matches':np.asarray(matches,int).reshape(-1,2).tolist()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',required=True);a=p.parse_args();B=a.root;E=B/'experiments/exp155_low_score_association_audit';O=B/'experiments/exp141_bytetrack_yolox_x_uavswarmv1';D=B/'datasets/UAVSwarm-dataset-master';R=E/'results'/a.run
 git=lambda *x:subprocess.check_output(['git','-C',str(ROOT),*x],text=True).strip()
 assert not git('status','--porcelain');cfg=json.loads((E/'configs/protocol.json').read_text());meta=json.loads((O/'results/run_002/detections/export_summary.json').read_text());gtmeta=json.loads((O/'results/run_003/metrics.json').read_text());names=sorted(x['sequence'] for x in meta['sequences']);assert len(names)==36;inputs={}
 for s in meta['sequences']:
  n=s['sequence'];f=O/'results/run_002/detections'/n/'det.txt';assert sha(f)==s['det_sha256'];inputs[str(f)]=sha(f);g=D/'test'/n/'gt/gt.txt';assert sha(g)==gtmeta['gt_sha256'][n];inputs[str(g)]=sha(g)
  for f in [D/'test'/n/'seqinfo.ini',O/'results/run_004/output/new_x/tracks/base'/f'{n}.txt']:inputs[str(f)]=sha(f)
 import motmetrics,scipy,trackeval
 R.mkdir(parents=True,exist_ok=False);(R/'tracks').mkdir();(R/'traces').mkdir();dump(R/'inputs.json',inputs)
 m={'status':'running','started_utc':now(),'private_repository':'git@github.com:YUSIO/ByteTrack-MOT.git','branch':'exp/155-low-score-association-audit','commit':git('rev-parse','HEAD'),'base_commit':'46820526c6477c3498ae596711df7fc9fb7633f6','code_dirty':False,'config_sha256':sha(E/'configs/protocol.json'),'input_manifest_sha256':sha(R/'inputs.json'),'python_executable':sys.executable,'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'motmetrics':motmetrics.__version__,'host':platform.node(),'device':'CPU','seed':155,'dataset':'UAVSwarmV1 canonical test 36seq/5754frames/37664GT','cache':'Exp141 run_002 YOLOX-X epoch80','evaluator':'unchanged homa.evaluate; union frames, CLEAR/ID IoU .5, TrackEval HOTA','visualization':'waived; diagnostic tables requested','artifact_recovery':'pending'};dump(R/'manifest.yaml',m);(R/'command.txt').write_text(shlex.join([sys.executable,*sys.argv])+'\n');code=0
 try:
  wrap_stracks(STrack);reg={}
  for n in names:
   info=seq_info(D/'test'/n);shape=(info['imHeight'],info['imWidth']);raw=np.loadtxt(O/'results/run_002/detections'/n/'det.txt',delimiter=',',ndmin=2);t=BYTETracker(SimpleNamespace(**cfg['tracker']),frame_rate=info['frameRate']);BaseTrack._count=0;fc=FCTrack(t,STrack,{});rows=[];record={}
   def cb(stage,ts,ds,cost,matches):record[stage]=capture(ts,ds,cost,matches)
   t.association_audit=cb
   with gzip.open(R/'traces'/f'{n}.jsonl.gz','wt') as log:
    for frame in range(1,info['seqLength']+1):
     r=raw[raw[:,0]==frame];det=np.column_stack((r[:,2:4],r[:,2:4]+r[:,4:6],r[:,6])).astype(np.float32);record={'frame':frame}
     for tid,box,score in fc.step(det,shape,lambda:None):
      x,y,w,h=box
      if w*h>0:rows.append('{},{},{},{},{},{},{},-1,-1,-1\n'.format(frame,tid,round(x,1),round(y,1),round(w,1),round(h,1),round(score,2)))
     log.write(json.dumps(record,separators=(',',':'))+'\n')
   text=''.join(rows);out=R/'tracks'/f'{n}.txt';out.write_text(text);ref=O/'results/run_004/output/new_x/tracks/base'/f'{n}.txt';assert out.read_bytes()==ref.read_bytes(),n;reg[n]={'frames':info['seqLength'],'rows':len(rows),'sha256':sha(out),'byte_identical':True};print(n,'byte_identical',flush=True)
  dump(R/'regression.json',reg);result=evaluate(D,'test',R/'tracks',R/'metrics.json',names);v=result['overall'];mm=v['motmetrics'];h=v['hota'];actual={**{k:h[k] for k in ['HOTA','DetA','AssA']},'MOTA':mm['mota'],'IDF1':mm['idf1'],'FP':mm['num_false_positives'],'FN':mm['num_misses'],'IDSW':mm['num_switches']};ref=json.loads((O/'results/run_004/metrics.json').read_text())['new_x']['arms']['base']['overall']
  for k,v in actual.items():assert abs(v-ref[k])<1e-5,(k,v,ref[k])
  print('all regression checks passed',actual,flush=True)
 except Exception:traceback.print_exc();code=1
 m.update(status='completed' if code==0 else 'failed',finished_utc=now(),exit_code=code);dump(R/'manifest.yaml',m);(R/'exit_code.txt').write_text(str(code)+'\n');(R/'report.md').write_text('# 决策日志重放\n\n'+('完成，待事后GT归因及回收。' if code==0 else '失败，保留此次尝试。')+'\n')
 (R/'output_checksums.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(R)}\n' for p in sorted(R.rglob('*')) if p.is_file() and p.name not in ['manifest.yaml','report.md','output_checksums.sha256','stdout.log','stderr.log']))
 sys.exit(code)
if __name__=='__main__':main()
