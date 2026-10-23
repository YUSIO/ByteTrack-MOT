"""Exp152: frozen V1 cache adaptation of unmodified upstream ARC-SORT."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,datetime,platform,traceback,shlex,copy,configparser,importlib.util,argparse
ROOT=Path(__file__).resolve().parent; V=ROOT/'vendor/arcsort_official';sys.path.insert(0,str(V));sys.path.insert(1,str(ROOT))
import numpy as np,cv2,torch,yaml,scipy,motmetrics
from loguru import logger
from arcsort.tracker import ARCTracker,load_config
from arcsort.cmc import CameraMotionCompensator
from arcsort.reid import EmbeddingComputer
from homa.evaluate import evaluate

def mod(name):
 s=importlib.util.spec_from_file_location('arc_'+name,V/'tools'/f'{name}.py');m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
rc=mod('run_cached');ce=mod('cache_embeddings');inter=mod('interpolate_tracks');merge=mod('merge_tracklets')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def git(*a):return subprocess.check_output(['git','-C',str(ROOT),*a],text=True).strip()
class Instrumented(ARCTracker):
 def __init__(self,c):
  super().__init__(c);self.audit={'episodes_opened':0,'episodes_closed':0,'evidence_frames':0,'negative_closed':0,'expansion_values':[]}
 def _arb_touch_episode(self,a,b):
  n=len(self.arb_episodes);super()._arb_touch_episode(a,b);self.audit['episodes_opened']+=len(self.arb_episodes)>n
 def _arbitrate(self,emb):
  old=copy.deepcopy(self.arb_episodes);super()._arbitrate(emb)
  for k,ep in old.items():
   if k not in self.arb_episodes:
    self.audit['episodes_closed']+=1;self.audit['evidence_frames']+=ep['n_evidence'];self.audit['negative_closed']+=ep['llr']<0
 def _compute_track_expansions(self,*a,**kw):
  v=super()._compute_track_expansions(*a,**kw)
  if v is not None:self.audit['expansion_values'].extend(np.asarray(v).flatten().tolist())
  return v

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['prepare','track'],required=True);p.add_argument('--run',required=True);a=p.parse_args();B=a.root;E=B/'experiments/exp152_arcsort_v1';R=E/'results'/a.run
 assert not git('status','--porcelain'), 'dirty code'
 cfg=json.loads((E/'configs/protocol.json').read_text());O=B/'experiments/exp141_bytetrack_yolox_x_uavswarmv1';D=B/'datasets/UAVSwarm-dataset-master';meta=json.loads((O/'results/run_002/detections/export_summary.json').read_text());gtmeta=json.loads((O/'results/run_003/metrics.json').read_text());names=sorted(s['sequence'] for s in meta['sequences']);assert len(names)==36
 inputs={};frames={};dets={}
 for s in meta['sequences']:
  n=s['sequence'];f=O/'results/run_002/detections'/n/'det.txt';assert sha(f)==s['det_sha256'];inputs[str(f)]=sha(f);dets[n]=rc.load_detections(f,min_confidence=.1)
  g=D/'test'/n/'gt/gt.txt';assert sha(g)==gtmeta['gt_sha256'][n];inputs[str(g)]=sha(g)
  ini=g.parent.parent/'seqinfo.ini';c=configparser.ConfigParser();c.read(ini);frames[n]=int(c['Sequence']['seqLength']);inputs[str(ini)]=sha(ini)
 assert sum(frames.values())==5754
 if a.phase=='prepare':
  assert torch.cuda.is_available()
  for n in names:
   imgs=sorted((D/'test'/n/'img1').glob('*.jpg'));assert len(imgs)==frames[n]
   for f in imgs:inputs[str(f)]=sha(f)
  weight=E/'assets/sports_model.pth.tar-60';assert sha(weight)==cfg['weights_sha256'];inputs[str(weight)]=sha(weight)
 else:
  P=E/'results/run_001';assert (P/'exit_code.txt').read_text().strip()=='0'
  for line in (P/'output_checksums.sha256').read_text().splitlines():
   h,f=line.split('  ',1);assert sha(P/f)==h
  for f in (P/'output').rglob('*'):
   if f.is_file():inputs[str(f)]=sha(f)
 R.mkdir(parents=True,exist_ok=False);(R/'output').mkdir();dump(R/'inputs.json',inputs)
 manifest={'status':'running','phase':a.phase,'started_utc':now(),'repository':'git@github.com:YUSIO/ByteTrack-MOT.git','branch':git('branch','--show-current'),'commit':git('rev-parse','HEAD'),'dirty':False,'upstream_arcsort':cfg['upstream_arcsort'],'config_sha256':sha(E/'configs/protocol.json'),'input_manifest_sha256':sha(R/'inputs.json'),'python_executable':sys.executable,'python':sys.version,'host':platform.node(),'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'opencv':cv2.__version__,'motmetrics':motmetrics.__version__,'seed':152,'dataset':'canonical UAVSwarmV1 test 36 sequences 5754 frames','cache':'Exp141 run_002 YOLOX-X epoch80','device':os.environ.get('CUDA_VISIBLE_DEVICES','CPU'),'evaluation':'pinned homa.evaluate; GT/result union; CLEAR/ID IoU .5; TrackEval HOTA','visualization':'waived; not requested','artifact_recovery':'pending'}
 dump(R/'manifest.yaml',manifest);(R/'command.txt').write_text(shlex.join([sys.executable,*sys.argv])+'\n');np.random.seed(152);torch.manual_seed(152);cv2.setRNGSeed(152);cv2.setNumThreads(1);torch.set_num_threads(2);logger.remove();logger.add(sys.stderr,level='WARNING')
 code=0
 try:
  base=load_config(str(V/'arcsort/configs/sportsmot.yaml'));base['output_box']='kalman';base['reid']['weights']=str(E/'results/run_001/output/sports_osnet_traced.pt')
  if a.phase=='prepare':
   clean=R/'output/sports_osnet_clean.pth';traced=R/'output/sports_osnet_traced.pt'
   for tool,args in [('extract_state_dict',['--input',str(weight),'--output',str(clean)]),('convert_osnet_weights',['--input',str(clean),'--output',str(traced)])]:
    subprocess.run([sys.executable,str(V/'tools'/f'{tool}.py'),*args],check=True)
   computer=EmbeddingComputer(str(traced),(128,256));cmc=CameraMotionCompensator(base['cmc']);audit={}
   (R/'output/cmc').mkdir();(R/'output/embeddings').mkdir()
   for n in names:
    cmc.reset();aff={}
    for t in range(1,frames[n]+1):
     f=D/'test'/n/'img1'/f'{t:06d}.jpg';image=cv2.imread(str(f));assert image is not None;aff[t]=cmc.apply(image)
    np.save(R/'output/cmc'/f'{n}.npy',aff,allow_pickle=True)
    out=R/'output/embeddings'/f'{n}.npz';count=ce.cache_sequence(computer,dets[n],D/'test'/n/'img1',out)
    z=np.load(out);expected=sum(len(x) for x in dets[n].values());assert count==expected
    for t,rows in dets[n].items():
     assert np.allclose(z[f'boxes_{t}'],rows[:,:4],atol=.001);em=z[f'embs_{t}'];assert np.isfinite(em).all() and (np.linalg.norm(em,axis=1)>.99).all()
    audit[n]={'frames':len(aff),'embeddings':count};dump(R/'metrics.json',audit);print(n,audit[n],flush=True)
  else:
   summary={};audits={};(R/'output/metrics').mkdir();P=E/'results/run_001/output'
   for box in ['kalman','detection']:
    for adaptive in [False,True]:
     for arb in [False,True]:
      arm=f'{box}_a{int(adaptive)}_r{int(arb)}';c=copy.deepcopy(base);c['eiou']['adaptive']=adaptive;c['eiou']['expansion']=.7;c['arbitration']['enabled']=arb;c['output_box']=box
      tracker=Instrumented(c);dump(R/(arm+'_config.json'),tracker.cfg)
      dirs={v:R/'output/tracks'/f'{arm}_{v}' for v in ['online','merge','interp','interp_merge','interp_splitmerge']}
      for d in dirs.values():d.mkdir(parents=True)
      aud={}
      for n in names:
       tracker=Instrumented(c);tracker.embedding_computer.load_cache(P/'embeddings'/f'{n}.npz');cmc=np.load(P/'cmc'/f'{n}.npy',allow_pickle=True).item();lines=[];events=[]
       for t in range(1,frames[n]+1):
        rows=dets[n].get(t,np.empty((0,6)));out=tracker.update(rows,frame=None,cmc_affine=cmc[t]);assert not tracker.embedding_computer._cache_miss_warned
        for x1,y1,x2,y2,tid in out:lines.append(f'{t},{int(tid)},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},1,-1,-1,-1')
        events.extend(tracker.pop_ambiguity_events())
       (dirs['online']/f'{n}.txt').write_text('\n'.join(lines)+('\n' if lines else ''))
       il=inter.interpolate_sequence(lines,max_gap=10);(dirs['interp']/f'{n}.txt').write_text('\n'.join(il)+('\n' if il else ''))
       v=tracker.audit.pop('expansion_values');stats={**tracker.audit,'expansion_count':len(v),'expansion_quantiles':np.percentile(v,[0,25,50,75,100]).tolist() if v else [],'ambiguity_count':len(events),'swaps':tracker.arbitration_swaps,'online_rows':len(lines),'interpolated_rows':len(il)-len(lines)}
       for variant,src,split in [('merge','online',False),('interp_merge','interp',False),('interp_splitmerge','interp',True)]:
        args=argparse.Namespace(split=split,split_thresh=.45,split_persist=5,merge_thresh=.85,max_gap=300,motion_tol=float('inf'))
        stats[variant]=merge.process_sequence(dirs[src]/f'{n}.txt',P/'embeddings'/f'{n}.npz',dirs[variant]/f'{n}.txt',args)
        # Offline stages may only change IDs; interpolated scores are serialized to 1 by upstream.
        def geometry(f):return sorted(tuple(float(y) for y in x.split(',')[:1]+x.split(',')[2:6]) for x in f.read_text().splitlines())
        assert geometry(dirs[src]/f'{n}.txt')==geometry(dirs[variant]/f'{n}.txt')
       for d in dirs.values():
        rows=[x.split(',') for x in (d/f'{n}.txt').read_text().splitlines()];assert len({(x[0],x[1]) for x in rows})==len(rows)
       aud[n]=stats
      for variant,d in dirs.items():
       key=arm+'_'+variant;m=evaluate(D,'test',d,R/'output/metrics'/f'{key}.json',names)['overall'];h=m['hota'];mm=m['motmetrics'];summary[key]={**{k:h[k] for k in ['HOTA','DetA','AssA']},'MOTA':mm['mota'],'IDF1':mm['idf1'],'IDSW':mm['num_switches'],'FP':mm['num_false_positives'],'FN':mm['num_misses']};dump(R/'metrics.json',summary);print(key,summary[key],flush=True)
      audits[arm]=aud;dump(R/'audit.json',audits)
 except Exception:
  traceback.print_exc();code=1
 manifest.update(status='completed' if code==0 else 'failed',finished_utc=now(),exit_code=code);dump(R/'manifest.yaml',manifest);(R/'exit_code.txt').write_text(str(code)+'\n');(R/'report.md').write_text('# 远端运行\n\n'+('完成，待回收核验。' if code==0 else '失败；保留日志，不覆盖重试。')+'\n')
 (R/'output_checksums.sha256').write_text(''.join(f'{sha(f)}  {f.relative_to(R)}\n' for f in sorted(R.rglob('*')) if f.is_file() and f.name not in ['manifest.yaml','report.md','output_checksums.sha256','stdout.log','stderr.log']))
 sys.exit(code)
if __name__=='__main__':main()
