"""Exact full-A query-cache replay with existing read-only decision tracer."""
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import sys,json,csv,hashlib,subprocess,datetime,traceback,platform,argparse,shlex,importlib.util
import numpy as np
from fctrack.run import seq_info
from fctrack.tracker import FCTrack,wrap_stracks
from fctrack.decision_trace import DecisionTrace
from yolox.tracker.byte_tracker import BYTETracker,STrack
from yolox.tracker.basetrack import BaseTrack
from homa.evaluate import evaluate
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--run',required=True);a=ap.parse_args();B=a.root;E=B/'experiments/exp156_full_a_decision_audit';D=B/'datasets/UAVSwarm-dataset-master';O=B/'experiments/exp141_bytetrack_yolox_x_uavswarmv1';R=E/'results'/a.run
 git=lambda *x:subprocess.check_output(['git','-C',str(ROOT),*x],text=True).strip()
 assert not git('status','--porcelain');cfg=json.loads((E/'configs/protocol.json').read_text());meta=json.loads((O/'results/run_002/detections/export_summary.json').read_text());gm=json.loads((O/'results/run_003/metrics.json').read_text());names=sorted(s['sequence'] for s in meta['sequences']);assert len(names)==36;inputs={}
 for s in meta['sequences']:
  n=s['sequence'];p=O/'results/run_002/detections'/n/'det.txt';assert sha(p)==s['det_sha256'];inputs[str(p)]=sha(p);g=D/'test'/n/'gt/gt.txt';assert sha(g)==gm['gt_sha256'][n];inputs[str(g)]=sha(g);i=D/'test'/n/'seqinfo.ini';inputs[str(i)]=sha(i)
 for rel,v in json.loads((E/'reference/provenance.json').read_text()).items():p=E/rel;assert sha(p)==v['sha256'];inputs[str(p)]=sha(p)
 import motmetrics,scipy
 R.mkdir(parents=True,exist_ok=False);(R/'tracks').mkdir();(R/'traces').mkdir();(R/'canonical').mkdir();dump(R/'inputs.json',inputs)
 m={'status':'running','started_utc':now(),'private_repository':'git@github.com:YUSIO/ByteTrack-MOT.git','branch':'exp/156-full-a-decision-audit','commit':git('rev-parse','HEAD'),'base_commit':'133989a48e5b91fafac31d7fca14a82b6f5123fd','code_dirty':False,'config_sha256':sha(E/'configs/protocol.json'),'input_manifest_sha256':sha(R/'inputs.json'),'python_executable':sys.executable,'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'motmetrics':motmetrics.__version__,'host':platform.node(),'device':'CPU; exact archived MPS verifier responses, no fresh inference','seed':156,'query_protocol':'kind/frame/order/crop allclose atol1e-8 rtol0; consume exactly once; full output byte identity required','visualization':'waived; diagnostic tables requested','artifact_recovery':'pending'};dump(R/'manifest.yaml',m);(R/'command.txt').write_text(shlex.join([sys.executable,*sys.argv])+'\n');code=0
 try:
  wrap_stracks(STrack);reg={}
  for n in names:
   info=seq_info(D/'test'/n);shape=(info['imHeight'],info['imWidth']);raw=np.loadtxt(O/'results/run_002/detections'/n/'det.txt',delimiter=',',ndmin=2);t=BYTETracker(SimpleNamespace(**cfg['tracker']),frame_rate=info['frameRate']);BaseTrack._count=0;fc=FCTrack(t,STrack,cfg['arms']['A'],log=True);q=defaultdict(list);used=defaultdict(int);err=[0.]
   for row in csv.DictReader((E/'reference/queries/A'/f'{n}.csv').open()):q[(row['kind'],int(row['frame']))].append(row)
   def cache(kind):
    def call(image,boxes):
     boxes=np.asarray(boxes,float).reshape(-1,4);key=(kind,fc.frame);start=used[key];rr=q[key][start:start+len(boxes)];assert len(rr)==len(boxes),(n,key,'missing query');ref=np.array([[float(r[k]) for k in ['x','y','w','h']] for r in rr]);difference=float(np.max(abs(boxes-ref))) if len(boxes) else 0.;err[0]=max(err[0],difference);assert np.allclose(boxes,ref,atol=1e-8,rtol=0),(n,key,difference);used[key]+=len(boxes)
     return np.array([float(r['p']) for r in rr]),np.array([[float(r[k]) for k in ['rx','ry','rw','rh']] for r in rr])
    return call
   fc.verify=cache('fill');fc.verify_birth=cache('birth');fc.trace=t.decision_trace=DecisionTrace(R/'traces'/f'{n}.jsonl.gz');rows=[]
   for frame in range(1,info['seqLength']+1):
    rr=raw[raw[:,0]==frame];det=np.column_stack((rr[:,2:4],rr[:,2:4]+rr[:,4:6],rr[:,6])).astype(np.float32)
    for tid,box,score in fc.step(det,shape,lambda:None):
     x,y,w,h=box
     if w*h>0:rows.append('{},{},{},{},{},{},{},-1,-1,-1\n'.format(frame,tid,round(x,1),round(y,1),round(w,1),round(h,1),round(score,2)))
   fc.trace.close();assert all(used[k]==len(v) for k,v in q.items()),n;out=R/'tracks'/f'{n}.txt';out.write_text(''.join(rows));assert out.read_bytes()==(E/'reference/tracks/A'/f'{n}.txt').read_bytes(),n;reg[n]={'byte_identical':True,'query_rows':sum(used.values()),'max_query_coordinate_error':err[0],'frames':info['seqLength'],'rows':len(rows)};print(n,reg[n],flush=True)
  dump(R/'regression.json',reg);result=evaluate(D,'test',R/'tracks',R/'metrics.json',names);ref=json.loads((E/'reference/metrics/A.json').read_text());assert result['overall']['motmetrics']==ref['overall']['motmetrics'],'CLEAR/ID regression failed'
  assert all(result['per_sequence'][n]['motmetrics']==ref['per_sequence'][n]['motmetrics'] for n in names),'per-sequence CLEAR/ID regression failed'
  dump(R/'metric_regression.json',{'clear_id_exact':True,'hota_exact':result==ref,'hota_overall_reference':ref['overall']['hota'],'hota_overall_replay':result['overall']['hota'],'boundary':'Identical output bytes; HOTA cross-runtime discrepancy retained, not a method gain.'})
  spec=importlib.util.spec_from_file_location('pinned',ROOT/'tools/evaluate_uavswarm_mot.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev);ev.mm.lap.default_solver='lap'
  for n in names:
   acc=ev.mm.utils.compare_to_groundtruth(ev.mm.io.loadtxt(str(D/'test'/n/'gt/gt.txt'),fmt='mot15-2D',min_confidence=1),ev.mm.io.loadtxt(str(R/'tracks'/f'{n}.txt'),fmt='mot15-2D',min_confidence=-1),'iou',distth=.5)
   acc.mot_events.reset_index().to_csv(R/'canonical'/f'{n}.csv',index=False)
  print('all query/output/metric regression gates passed',flush=True)
 except Exception:traceback.print_exc();code=1
 m.update(status='completed' if code==0 else 'failed',finished_utc=now(),exit_code=code);dump(R/'manifest.yaml',m);(R/'exit_code.txt').write_text(str(code)+'\n');(R/'report.md').write_text('# 完整A日志重放\n\n'+('完成，待归因。' if code==0 else '失败，保留此次尝试。')+'\n')
 (R/'output_checksums.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(R)}\n' for p in sorted(R.rglob('*')) if p.is_file() and p.name not in ['manifest.yaml','report.md','output_checksums.sha256','stdout.log','stderr.log']))
 sys.exit(code)
if __name__=='__main__':main()
