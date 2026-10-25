"""Controlled SCOR geometry migration; common birth interface, distance gate and ID memory.

Paper: Yoon et al., IJCV 2019, Eq.21,26-28. No SCEA or appearance reproduction.
State [rx, ry, vx, vy]; standard symmetric CV Q corrects the printed Q layout.
P0 is explicitly experimental (not specified in paper); sensitivity is preregistered.
"""
import numpy as np
from fctrack.tracker import FCTrack, iou_matrix
from fctrack.relink import carry
from fctrack.relabel_reliability import form_spread, saved_centre
from fctrack.relabel_probability import evidence

class RelativeEdges:
    def __init__(self, p0=9.):
        self.edges = {}
        self.frame = 0
        self.p0 = p0
        self.F = np.eye(4); self.F[0,2] = self.F[1,3] = 1.
        G = np.array([[.5,0],[0,.5],[1,0],[0,1.]])
        self.Q = G @ G.T  # sigma_sc=1, order x,y,vx,vy
        self.H = np.eye(4)[:2]; self.R = np.eye(2)*9.
    def predict(self, frame):
        assert frame >= self.frame
        for _ in range(frame-self.frame):
            for e in self.edges.values():
                e[0] = self.F @ e[0]
                e[1] = self.F @ e[1] @ self.F.T + self.Q
        self.frame = frame
    def observe(self, centres, lastw, gap):
        for key in list(self.edges):
            if any(k not in lastw or self.frame-lastw[k][0]>gap for k in key):
                del self.edges[key]
        for i in centres:
            for j in centres:
                if i == j: continue
                z = centres[i]-centres[j]
                key = (i,j)
                if key not in self.edges:
                    self.edges[key] = [np.r_[z,0.,0.], np.eye(4)*self.p0]
                else:
                    x,P = self.edges[key]
                    K = np.linalg.solve(self.H@P@self.H.T+self.R, self.H@P).T
                    x = x+K@(z-self.H@x)
                    A = np.eye(4)-K@self.H
                    P = A@P@A.T+K@self.R@K.T
                    self.edges[key] = [x,(P+P.T)/2]
    def estimate(self, label, current, reference, c1, offset=False):
        peers = [j for j in current if (label,j) in self.edges and (not offset or j in reference)]
        if not peers: return c1.copy(), None, 0
        j = min(peers,key=lambda j:(np.linalg.norm(self.edges[label,j][0][2:]),j))
        r = c1-reference[j] if offset else self.edges[label,j][0][:2]
        return current[j]+r, int(j), len(peers)

class ScorRelabelTrack(FCTrack):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.relative = RelativeEdges(self.relab.get("p0",9.))
        self.lastbox = {}
        self.relabel_events = []
        self.reliability_history = {}
    def step(self,*args,**kwargs):
        self.relative.predict(self.frame+1)
        rows = super().step(*args,**kwargs)
        centres = {tid:b[:2]+b[2:]/2 for tid,b,_ in rows if b[2]*b[3]>0}
        self.relative.observe(centres,self.lastw,self.relab["gap"])
        for tid,b,_ in rows:
            if b[2]*b[3]>0:self.lastbox[tid]=b.copy()
        self.reliability_history[self.frame] = {tid:saved_centre(b) for tid,b,_ in rows if b[2]*b[3]>0}
        for f in list(self.reliability_history):
            if f < self.frame-self.relab['gap']-1:
                del self.reliability_history[f]
        return rows
    def prediction(self,family,c1,side1,f1,f,L,cur):
        ref = self.hist.get(f1,{})
        if family in ("scor","scor_offset"):
            return self.relative.estimate(L,cur,ref,c1,family=="scor_offset")
        p = carry(family,c1,side1,f1,f,L,{f1:ref,f:cur},self.warps)
        return p,None,len(set(ref)&set(cur)-{L})
    def relabel_births(self,f,rows):
        r=self.relab
        births=[k for k,row in enumerate(rows) if row[0] not in self.lastw and row[1][2]*row[1][3]>0]
        if not births or f==1:return
        now={row[0] for row in rows}
        cur={row[0]:row[1][:2]+row[1][2:]/2 for row in rows if row[0] in self.lastw}
        relcur={row[0]:saved_centre(row[1]) for row in rows if row[0] in self.lastw}
        lost={t.track_id for t in self.t.lost_stracks}
        removed={t.track_id for t in self.t.removed_stracks}
        pairs=[];logs=[]
        for i,k in enumerate(births):
            b=rows[k][1];c0=b[:2]+b[2:]/2;side=float(np.sqrt(max(b[2]*b[3],1.)))
            e=dict(frame=f,new_id=int(rows[k][0]),box=b.tolist(),candidates=[],chosen=None,pre_veto_chosen=None,reliability_rejected=False)
            logs.append(e)
            for L,(f1,c1,side1) in self.lastw.items():
                if L in now or not 1<=f-f1<=r["gap"]:continue
                predictions={}
                for family in ("form","scor","scor_offset","abs","gmc"):
                    p,anchor,n=self.prediction(family,c1,side1,f1,f,L,cur)
                    bb=np.r_[p-self.lastbox[L][2:]/2,self.lastbox[L][2:]]
                    predictions[family]=dict(centre=p.tolist(),distance=float(np.linalg.norm(p-c0))/side,
                        iou=float(iou_matrix([bb],[b])[0,0]),anchor=anchor,peers=n)
                pred=predictions[r["family"]];d=pred["distance"]
                spread, npeer = form_spread(L,c1,side1,self.reliability_history.get(f1,{}),relcur)
                admissible=d<=r["tau"] and (r.get("gate","distance")=="distance" or pred["iou"]>np.exp(-4))
                e["candidates"].append(dict(id=int(L),last_frame=f1,gap=f-f1,last_box=self.lastbox[L].tolist(),
                    pool="Lost" if L in lost else "Removed" if L in removed else "history",predictions=predictions,admissible=bool(admissible),form_spread=spread,form_spread_peers=npeer))
                if admissible:pairs.append((d,i,L))
        done_i,done_L=set(),set()
        for d,i,L in sorted(pairs):
            if i in done_i or L in done_L:continue
            done_i.add(i);done_L.add(L)
            logs[i]['pre_veto_chosen'] = int(L)
            candidate = next(c for c in logs[i]['candidates'] if c['id']==L)
            probability = r.get('probability_veto')
            if probability is not None:
                assert r.get('spread_veto_threshold') is None
                result = evidence(d,candidate['form_spread'],probability)
                logs[i]['probability_evidence'] = result
                if result['rejected']:
                    logs[i]['reliability_rejected'] = True
                    self.stats['relabel_probability_vetoed'] += 1
                    continue
            threshold = r.get('spread_veto_threshold')
            if threshold is not None and candidate['form_spread'] is not None and candidate['form_spread'] > threshold:
                # Veto the original greedy assignment; do not retry another ID
                # or reallocate the reserved ID to a different birth this frame.
                logs[i]['reliability_rejected'] = True
                self.stats['relabel_spread_vetoed'] += 1
                continue
            row=rows[births[i]]
            self.t.lost_stracks=[x for x in self.t.lost_stracks if x.track_id!=L]
            self.t.removed_stracks=[x for x in self.t.removed_stracks if x.track_id!=L]
            old=row[0];row[3].track_id=L;row[0]=L
            if old in self.meta:
                m=self.meta.pop(old)
                self.meta[L]={**self.meta.get(L,{}),**{k:v for k,v in m.items() if k in ("real","box","prov")},
                    "obs":self.meta.get(L,{}).get("obs",0)+m.get("obs",0)}
            for snap in self.snap.values():
                if old in snap:snap[L]=snap.pop(old)
            self.stats["relabelled"]+=1
            logs[i]["chosen"]=int(L)
        self.relabel_events.extend(logs)
