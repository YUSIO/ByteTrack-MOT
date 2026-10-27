"""Exp192 natural U0 candidates, causal crop collection, current raw-box labels."""
from pathlib import Path
from collections import Counter
import argparse,csv,gzip,json,sys
import numpy as np
from scipy.optimize import linear_sum_assignment
SOURCE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(SOURCE))
from temporal_birth_common import Run,read,save
from fctrack.temporal_birth import ClipBuilder
from fctrack.tracker import iou_matrix

def records(path):
    with gzip.open(path,'rt') as stream:
        for line in stream:yield json.loads(line)

def strong_reference(trace,gt):
    seen=set();result={}
    for record in records(trace):
        f=record['frame'];events=record['events'];final=next(z for z in events if z['stage']=='final')['rows'];mapping={}
        for event in events:
            stage=event['stage']
            if stage in ('association_high','association_low','association_unconfirmed'):
                for i,j in event['matches']:mapping[event['tracks'][i]['uid']]=int(event['detection_indices'][j])
            if stage=='birth_decision' and event['decision']=='written':mapping[event['track']['uid']]=int(event['detection_index'])
            if stage=='byte_birth' and event['decision']=='activated':mapping[event['track']['uid']]=int(event['detection_index'])
        g=gt[gt[:,0]==f];boxes=np.asarray([z['tlwh'] for z in final],float).reshape(-1,4)
        ov=iou_matrix(boxes,g[:,2:6]);correct={}
        if ov.size:
            ii,jj=linear_sum_assignment(-(ov*(ov>=.5)));correct={i:int(g[j,1]) for i,j in zip(ii,jj) if ov[i,j]>=.5}
        for i,z in enumerate(final):
            if z['uid'] not in seen and i in correct and float(z['score'])>=.6 and z['uid'] in mapping:result[f,mapping[z['uid']]]=correct[i]
            seen.add(z['uid'])
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--experiment',default='experiments/exp192_temporal_birth_gate');p.add_argument('--config',default='configs/stage2.json');p.add_argument('--run',default='run_006');a=p.parse_args()
    with Run(a.root,a.experiment,a.config,a.run,'causal_candidate_collection') as run:
        root=run.root;cfg=run.cfg;plan=cfg['plan'];assert read(run.e/'analysis/budget_admission.json')['passed']
        import cv2,torch
        cv2.setNumThreads(1);torch.set_num_threads(1)
        allrows=[];clips=[];metadata=[];ref_counts={}
        scene={z['sequence']:z['scene_key'] for z in csv.DictReader((root/plan['data']['split_source']).open())}
        for group in ['fit','calibration']:
            for seq in plan['data']['groups'][group]:
                directory=root/plan['data']['dataset_root']/'train'/seq;fullgt=np.loadtxt(directory/'gt/gt.txt',delimiter=',',ndmin=2);gt=fullgt[fullgt[:,6]>=1]
                raw=np.loadtxt(root/plan['data']['detections']/seq/'det.txt',delimiter=',',ndmin=2);builder=ClipBuilder(directory,raw)
                references=strong_reference(run.e/'results/run_005/decisions'/f'{seq}.jsonl.gz',gt);comparable=set()
                for record in records(run.e/'results/run_002/decisions'/f'{seq}.jsonl.gz'):
                    f=record['frame'];g=gt[gt[:,0]==f]
                    event=next(z for z in record['events'] if z['stage']=='unified_candidates')
                    taken=np.asarray([z['tlwh'] for z in event['prebirth_rows']],float).reshape(-1,4);covered=set();ov=iou_matrix(g[:,2:6],taken)
                    if ov.size:
                        ii,jj=linear_sum_assignment(-(ov*(ov>=.5)));covered={int(g[i,1]) for i,j in zip(ii,jj) if ov[i,j]>=.5}
                    for index,box in zip(event['detection_indices'],event['boxes']):
                        index=int(index);np.testing.assert_allclose(builder.frames[f][index,:4],box,rtol=0,atol=1e-5)
                        eligible=[i for i,value in enumerate(iou_matrix([box],g[:,2:6])[0]) if value>=.5 and int(g[i,1]) not in covered]
                        values=iou_matrix([box],g[:,2:6])[0];owner=int(g[max(eligible,key=lambda i:values[i]),1]) if eligible else -1;label=int(bool(eligible))
                        ignored=fullgt[(fullgt[:,0]==f)&(fullgt[:,6]<1)]
                        if len(ignored) and np.any(iou_matrix([box],ignored[:,2:6])>=.5):label=-1
                        x,m=builder.clip(f,index);score=float(builder.frames[f][index,4]);is_ref=(f,index) in references
                        if is_ref:comparable.add((f,index))
                        allrows.append(dict(group=group,sequence=seq,scene=scene[seq],frame=f,index=index,owner=owner,label=label,score=score,band='weak' if score<.6 else 'strong',strong_reference=int(is_ref),x=box[0],y=box[1],w=box[2],h=box[3]))
                        clips.append(x);metadata.append(m)
                ref_counts[seq]=dict(original_A_correct_strong_events=len(references),comparable=len(comparable),not_comparable=len(references)-len(comparable))
                print('collected',group,seq,'cumulative',len(allrows),flush=True)
        assert len(allrows)>0
        with (run.r/'samples.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
        np.savez_compressed(run.r/'samples.npz',clips=np.stack(clips),metadata=np.stack(metadata),labels=np.array([z['label'] for z in allrows],np.int64))
        summary={};gates={}
        for group in ['fit','calibration']:
            summary[group]={}
            for band in ['all','weak','strong']:
                rows=[z for z in allrows if z['group']==group and (band=='all' or z['band']==band)];pos=[z for z in rows if z['label']==1]
                summary[group][band]=dict(rows=len(rows),positive=len(pos),negative=sum(z['label']==0 for z in rows),ignored=sum(z['label']<0 for z in rows),positive_unique_GTframes=len({(z['sequence'],z['frame'],z['owner']) for z in pos}),positive_scenes=len({z['scene'] for z in pos}))
            for band in ['all','weak']:
                z=summary[group][band];minimum=30 if group=='fit' and band=='all' else 10
                gates[group+'_'+band]=z['positive_unique_GTframes']>=minimum and z['positive_scenes']>=3 and z['negative']>0
        result=dict(groups=summary,reference_counts=ref_counts,G0_gates=gates,G0_passed=all(gates.values()),samples=len(allrows))
        save(run.r/'metrics.json',result);run.manifest.update(device='cpu',neural_model_executed=False,training=False,calibration_read=True,versions={'numpy':np.__version__,'torch':torch.__version__,'opencv':cv2.__version__});print(json.dumps(result,indent=2))
if __name__=='__main__':main()
