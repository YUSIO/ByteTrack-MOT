"""GT-assisted constrained closed-loop actions; no GT coordinates become proposals."""
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict,Counter
import argparse,copy,csv,gzip,hashlib,json,subprocess,sys,time,platform,datetime,traceback
import numpy as np
import cv2,torch
from fctrack.tracker import FCTrack,wrap_stracks,iou_matrix
from fctrack.relink import carry
from fctrack.run import seq_info
from fctrack.verifier import load,verify
from yolox.tracker.byte_tracker import BYTETracker,STrack
from yolox.tracker.basetrack import BaseTrack,TrackState
from homa.evaluate import evaluate
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
def label(b,g):
 ix=np.flatnonzero(iou_matrix([b],g[:,2:6])[0]>=.5);return int(g[ix[0],1]) if len(ix)==1 else None
def pred(t):
 m=t.mean.copy()
 if t.state!=TrackState.Tracked:m[7]=0
 return t.kalman_filter.predict(m,t.covariance)
def boxmean(m):
 b=m[:4].copy();b[2]*=b[3];b[:2]-=b[2:]/2;return b
class AuditA(FCTrack):
 def relabel_births(self,f,rows):
  if self.arm not in ['identity','joint']:return super().relabel_births(f,rows)
  r=self.relab;births=[k for k,row in enumerate(rows) if row[0] not in self.lastw and row[1][2]*row[1][3]>0]
  if not births or f==1:return
  now={row[0] for row in rows};cur={row[0]:row[1][:2]+row[1][2:]/2 for row in rows if row[0] in self.lastw};pairs=[]
  for i,k in enumerate(births):
   b=rows[k][1];c=b[:2]+b[2:]/2;side=np.sqrt(max(b[2]*b[3],1.));target=label(b,self.g)
   for L,(f1,c1,s1) in self.lastw.items():
    if L in now or not 1<=f-f1<=r['gap']:continue
    p=carry(r['family'],c1,s1,f1,f,L,{f1:self.hist.get(f1,{}),f:cur},self.warps);d=float(np.linalg.norm(p-c))/side
    if d>r['tau']:continue
    h=self.public.get(L,[]);known=len(h)>=3 and h[-1][1] is not None and all(z[1]==h[-1][1] for z in h[-3:]);status='unknown'
    if known and target is not None:status='correct' if h[-1][1]==target else 'wrong'
    self.events.append(dict(frame=f,kind='identity_edge',old=L,new=rows[k][0],distance=d,status=status))
    if status=='wrong':self.count['identity_edges_vetoed']+=1;continue
    pairs.append((0 if status=='correct' else 1,d,i,L))
  di,dl=set(),set()
  for _,d,i,L in sorted(pairs):
   if i in di or L in dl:continue
   di.add(i);dl.add(L);row=rows[births[i]];old=row[0]
   self.t.lost_stracks=[x for x in self.t.lost_stracks if x.track_id!=L];self.t.removed_stracks=[x for x in self.t.removed_stracks if x.track_id!=L];row[3].track_id=L;row[0]=L
   if old in self.meta:
    m=self.meta.pop(old);self.meta[L]={**self.meta.get(L,{}),**{k:v for k,v in m.items() if k in ('real','box','prov')},'obs':self.meta.get(L,{}).get('obs',0)+m.get('obs',0)}
   for snap in self.snap.values():
    if old in snap:snap[L]=snap.pop(old)
   self.stats['relabelled']+=1;self.count['identity_relinks']+=1

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--run',required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--protocol',default='actions.json');a=ap.parse_args();B=a.root;E=B/'experiments/exp157_mechanism_gain_budget';W=Path(__file__).resolve().parent;R=E/('preflight' if a.smoke else 'results')/a.run
 git=lambda *x:subprocess.check_output(['git','-C',str(W),*x],text=True).strip()
 assert not git('status','--porcelain');cfg=json.loads((E/'configs/base.json').read_text());prot=json.loads((E/'configs'/a.protocol).read_text());cell=cfg['cells']['new_x'];D=B/cell['dataset_root']/'test';cache=B/cell['detections'];REF=B/'experiments/exp156_full_a_decision_audit/reference';names=sorted(p.name for p in cache.glob('UAVSwarm-*'));names=names[:2] if a.smoke else names;assert a.smoke or len(names)==36
 torch.set_num_threads(1);torch.manual_seed(157);np.random.seed(157);cv2.setNumThreads(1);device=torch.device('cuda:1');assert torch.cuda.is_available();weight=B/cell['verifiers']['seed']['weights'];model=load('seed',weight,device);model.eval()
 inputs={str(p.relative_to(B)):sha(p) for p in [weight,E/'configs/base.json',E/'configs'/a.protocol,W/'mechanism_budget.py']}
 detmeta=json.loads((cache/'export_summary.json').read_text());gtmeta=json.loads((B/'experiments/exp141_bytetrack_yolox_x_uavswarmv1/results/run_003/metrics.json').read_text())
 for n in names:
  dp=cache/n/'det.txt';assert sha(dp)==next(s['det_sha256'] for s in detmeta['sequences'] if s['sequence']==n);assert sha(D/n/'gt/gt.txt')==gtmeta['gt_sha256'][n]
  for p in [dp,D/n/'gt/gt.txt',D/n/'seqinfo.ini',REF/'queries/A'/f'{n}.csv',REF/'tracks/A'/f'{n}.txt']:inputs[str(p.relative_to(B))]=sha(p)
 R.mkdir(parents=True,exist_ok=False);dump(R/'inputs.json',inputs);(R/'command.txt').write_text(' '.join(sys.argv)+'\n');manifest={'status':'running','commit':git('rev-parse','HEAD'),'clean':True,'base':'fc2bb11246f30edda20167a46750208ab1f3e917','branch':'exp/157-mechanism-budget','private_repository':'git@github.com:YUSIO/ByteTrack-MOT.git','python_executable':sys.executable,'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'host':platform.node(),'device':str(device),'GT_used_in_decisions':True,'seed':157,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol':prot,'recovery':'pending','smoke':a.smoke};dump(R/'manifest.yaml',manifest);rc=0;summary={}
 try:
  wrap_stracks(STrack)
  for arm in prot['arms']:
   AR=R/arm;tracks=AR/'tracks';tracks.mkdir(parents=True);stats={};start=time.time()
   for n in names:
    info=seq_info(D/n);shape=(info['imHeight'],info['imWidth']);raw=np.loadtxt(cache/n/'det.txt',delimiter=',',ndmin=2);gt=np.loadtxt(D/n/'gt/gt.txt',delimiter=',',ndmin=2);gt=gt[gt[:,6]>=1];refs=defaultdict(list)
    for x in csv.DictReader((REF/'queries/A'/f'{n}.csv').open()):refs[(x['kind'],int(x['frame']))].append(x)
    fc=AuditA(BYTETracker(SimpleNamespace(**cfg['tracker']),frame_rate=info['frameRate']),STrack,copy.deepcopy(cfg['arms']['A']));BaseTrack._count=0;fc.arm=arm;fc.public=defaultdict(list);fc.events=[];fc.count=Counter();owners=defaultdict(list);templates={};rowsout=[];frameimage=[None];framegray=[None];imagehash={}
    def image():
     if frameimage[0] is None:
      ip=D/n/'img1'/f'{fc.frame:06d}.jpg';frameimage[0]=cv2.imread(str(ip));assert frameimage[0] is not None,ip;imagehash[str(ip.relative_to(B))]=sha(ip)
     return frameimage[0]
    def gray():
     if framegray[0] is None:framegray[0]=cv2.cvtColor(image(),cv2.COLOR_BGR2GRAY)
     return framegray[0]
    def call(kind):
     def run(_,boxes):
      boxes=np.asarray(boxes,float).reshape(-1,4);pp=np.empty(len(boxes));bb=np.empty_like(boxes);miss=[]
      for j,b in enumerate(boxes):
       found=None
       for x in refs.get((kind,fc.frame),[]):
        z=np.array([float(x[k]) for k in ['x','y','w','h']])
        if np.allclose(b,z,atol=1e-8,rtol=0):found=x;break
       if found:
        pp[j]=float(found['p']);bb[j]=[float(found[k]) for k in ['rx','ry','rw','rh']];fc.count['archived_queries']+=1
       else:miss.append(j)
      if miss:
       p,b=verify('seed',model,image(),boxes[miss],device);pp[miss]=p;bb[miss]=b;fc.count['fresh_queries']+=len(miss)
      return pp,bb
     return run
    fc.verify=call('fill');fc.verify_birth=call('birth')
    for f in range(1,info['seqLength']+1):
     frameimage[0]=framegray[0]=None;g=gt[gt[:,0]==f];fc.g=g;before={}
     for t in fc.t.tracked_stracks+fc.t.lost_stracks:
      if t.mean is None or not t.is_activated:continue
      h=owners[t];stable=len(h)>=3 and h[-1][1] is not None and all(z[1]==h[-1][1] for z in h[-3:]);before[t]={'owner':h[-1][1] if stable else None,'pred':pred(t),'meta':copy.deepcopy(fc.meta.get(t.track_id)),'box':getattr(t,'_det',t.tlwh).copy(),'fid':t.frame_id}
     rr=raw[raw[:,0]==f];det=np.column_stack((rr[:,2:4],rr[:,2:4]+rr[:,4:6],rr[:,6])).astype(np.float32);rows=fc.step(det,shape,lambda:None);objects={t.track_id:t for t in fc.t.tracked_stracks};rowmap={pid:j for j,(pid,_,_) in enumerate(rows)};guarded=set();locupdated=set()
     # Keep current outputs/lifecycle; only the future KF and FCTrack measurement memory are protected.
     if arm in ['state','joint']:
      for t,old in before.items():
       if t.frame_id!=f or t.track_id not in rowmap or old['owner'] is None or old['meta'] is None:continue
       tg=g[g[:,1]==old['owner']]
       if len(tg)!=1:continue
       b=np.asarray(rows[rowmap[t.track_id]][1]);ov=float(iou_matrix([b],tg[:,2:6])[0,0])
       if ov>=.5:continue
       t.mean,t.covariance=old['pred'][0].copy(),old['pred'][1].copy();t._det=old['box'].copy();fc.meta[t.track_id]=copy.deepcopy(old['meta']);fc.snap.get(f,{}).pop(t.track_id,None);guarded.add(t);fc.count['state_writes_blocked']+=1;fc.events.append(dict(frame=f,kind='state_guard',id=t.track_id,owner=old['owner'],iou=ov))
     if arm in ['location','location_quality','location_native','location_quality_current','location_current_native','location_quality_no_corr','joint']:
      seeds=[];mapping=[];eligible=[]
      for t,old in before.items():
       j=rowmap.get(t.track_id);m=fc.meta.get(t.track_id)
       if j is not None and objects.get(t.track_id) is not t:
        fc.count['relabel_replaced_root_skipped']+=1;continue
       if j is None and not(t in fc.t.lost_stracks and t.state==TrackState.Lost and m and 1<=f-t.frame_id<=5 and f-m['real']<=30):continue
       eligible.append(t);b=np.asarray(rows[j][1] if j is not None else t.tlwh,float);pb=boxmean(old['pred'][0]);candidates=[('current',b)] if arm in ['location_quality_current','location_current_native'] else [('current',b),('prediction',pb)]
       if t in templates and arm not in ['location_quality_current','location_current_native','location_quality_no_corr']:
        template=templates[t];ph,pw=template.shape;center=pb[:2]+pb[2:]/2;radius=max(16.,2*np.sqrt(max(pb[2]*pb[3],1.)));x0=max(0,int(center[0]-radius-pw/2));y0=max(0,int(center[1]-radius-ph/2));x1=min(shape[1],int(center[0]+radius+pw/2)+1);y1=min(shape[0],int(center[1]+radius+ph/2)+1);search=gray()[y0:y1,x0:x1]
        if search.shape[0]>=ph and search.shape[1]>=pw and template.std()>1e-6:
         corr=cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED);_,_,_,xy=cv2.minMaxLoc(corr);cb=np.array([x0+xy[0]+pw/2-pb[2]/2,y0+xy[1]+ph/2-pb[3]/2,pb[2],pb[3]]);candidates.append(('correlation',cb))
       for origin,b in candidates:
        if np.isfinite(b).all() and min(b[2:])>1:seeds.append(b);mapping.append((t,origin))
      proposals=defaultdict(list)
      if seeds:
       pp,bb=verify('seed',model,image(),np.asarray(seeds),device);fc.count['local_search_queries']+=len(seeds)
       for (t,origin),p,b in zip(mapping,pp,bb):
        if p>=.97 and np.isfinite(b).all() and min(b[2:])>1:proposals[t].append((float(p),b,origin))
      # Candidates were generated for all eligible tracks without GT; selection below is assisted.
      for t in eligible:
       old=before[t];j=rowmap.get(t.track_id);tg=g[g[:,1]==old['owner']] if old['owner'] is not None else np.empty((0,gt.shape[1]));qs=proposals[t];fc.count['local_candidates_passed']+=len(qs)
       if arm not in ['location_native','location_current_native']:
        if len(tg)!=1:continue
        currentiou=float(iou_matrix([rows[j][1]],tg[:,2:6])[0,0]) if j is not None else 0.
        if arm not in ['location_quality','location_quality_current','location_quality_no_corr'] and currentiou>=.5:continue
        qs=[x for x in qs if label(x[1],g)==old['owner'] and (arm not in ['location_quality','location_quality_current','location_quality_no_corr'] or float(iou_matrix([x[1]],tg[:,2:6])[0,0])>currentiou+1e-9)]
        qs.sort(key=lambda x:float(iou_matrix([x[1]],tg[:,2:6])[0,0]),reverse=True)
       else:qs.sort(key=lambda x:x[0],reverse=True)
       for p,b,origin in qs:
        other=np.asarray([z[1] for k,z in enumerate(rows) if k!=j]).reshape(-1,4)
        if len(other) and iou_matrix([b],other).max()>=.3:continue
        if j is None:
         t.re_activate(STrack(b,.7),f,new_id=False);fc.t.lost_stracks=[z for z in fc.t.lost_stracks if z is not t];fc.t.tracked_stracks.append(t);rows.append((t.track_id,b.copy(),.7));rowmap[t.track_id]=len(rows)-1
        else:
         t.mean,t.covariance=t.kalman_filter.update(t.mean,t.covariance,t.tlwh_to_xyah(b));t._det=b.copy();rows[j]=(t.track_id,b.copy(),rows[j][2])
        if t.track_id in fc.meta:fc.meta[t.track_id]['box']=b.copy()
        fc.snap.setdefault(f,{})[t.track_id]=b[:2]+b[2:]/2;fc.hist.setdefault(f,{})[t.track_id]=b[:2]+b[2:]/2;fc.lastw[t.track_id]=(f,b[:2]+b[2:]/2,float(np.sqrt(b[2]*b[3])));locupdated.add(t);fc.count['location_actions']+=1;fc.events.append(dict(frame=f,kind='location',id=t.track_id,owner=old['owner'],origin=origin,p=p,was_missing=j is None));break
     # Prior identities use only actual detector observations or initial verifier birth, never recurrent fills.
     for t in fc.t.tracked_stracks:
      if t in guarded:continue
      m=fc.meta.get(t.track_id)
      if t.frame_id==f and m and m['real']==f:
       b=getattr(t,'_det',t.tlwh);owners[t].append((f,label(b,g)))
       if arm in ['location','location_quality','location_native','location_quality_current','location_current_native','location_quality_no_corr','joint']:
        x,y,w,h=b;xa=max(0,int(x));ya=max(0,int(y));xb=min(shape[1],int(np.ceil(x+w)));yb=min(shape[0],int(np.ceil(y+h)))
        if xb-xa>=2 and yb-ya>=2:templates[t]=gray()[ya:yb,xa:xb].copy()
     for pid,b,score in rows:
      fc.public[pid].append((f,label(b,g)));x,y,w,h=b
      if w*h>0:rowsout.append('{},{},{},{},{},{},{},-1,-1,-1\n'.format(f,pid,round(x,1),round(y,1),round(w,1),round(h,1),round(score,2)))
    out=tracks/f'{n}.txt';out.write_text(''.join(rowsout))
    if arm=='A':assert out.read_bytes()==(REF/'tracks/A'/f'{n}.txt').read_bytes(),n;assert fc.count['fresh_queries']==0
    with gzip.open(AR/f'{n}.events.json.gz','wt') as h:json.dump(fc.events,h)
    dump(AR/f'{n}.images.json',imagehash);stats[n]=dict(fc.count);print(arm,n,dict(fc.count),flush=True)
   metrics=evaluate(B/cell['dataset_root'],'test',tracks,AR/'metrics.json',names);metrics['protocol']['gt_used_only_for_evaluation']=arm in ['A','location_native','location_current_native'];dump(AR/'metrics.json',metrics);dump(AR/'stats.json',stats);summary[arm]={'overall':metrics['overall'],'seconds':time.time()-start,'counts':dict(sum((Counter(v) for v in stats.values()),Counter()))};dump(R/'summary.json',summary)
  assert summary['A']['overall']['motmetrics']==json.loads((REF/'metrics/A.json').read_text())['overall']['motmetrics'] or a.smoke
 except Exception:traceback.print_exc();rc=1
 manifest.update(status='completed' if rc==0 else 'failed',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=rc);dump(R/'manifest.yaml',manifest);(R/'exit_code.txt').write_text(str(rc)+'\n');(R/'report.md').write_text('# 运行记录\n\n'+('完成，待收益判读。' if rc==0 else '失败保留。')+'\n');(R/'payload_checksums.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(R))+'\n' for p in sorted(R.rglob('*')) if p.is_file() and p.name not in ['payload_checksums.sha256','manifest.yaml','report.md']));sys.exit(rc)
if __name__=='__main__':main()
