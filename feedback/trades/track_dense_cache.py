"""CPU replay of frozen dense-gate caches; no detector execution or tuning."""
import argparse
from collections import defaultdict
import datetime
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from trackeval.metrics import CLEAR, Identity
from trackeval.datasets._base_dataset import _BaseDataset
from feedback.data import sequence_info
from .history_counterfactual import HistoryReplay, cache_detections


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def evaluator(path):
    spec = importlib.util.spec_from_file_location('pinned_mot_eval', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.mm.lap.default_solver = 'lap'
    return module


def official_data(g, p, length):
    """Same full-frame TrackEval input convention as Exp167 CLEAR audit."""
    gm = {v:i for i,v in enumerate(sorted(set(g[:,1].astype(int))))}
    pm = {v:i for i,v in enumerate(sorted(set(p[:,1].astype(int))))}
    d = dict(gt_ids=[], tracker_ids=[], similarity_scores=[], num_timesteps=length,
             num_gt_ids=len(gm), num_tracker_ids=len(pm), num_gt_dets=len(g), num_tracker_dets=len(p))
    for frame in range(1, length+1):
        gg, pp = g[g[:,0]==frame], p[p[:,0]==frame]
        d['gt_ids'].append(np.array([gm[int(v)] for v in gg[:,1]], int))
        d['tracker_ids'].append(np.array([pm[int(v)] for v in pp[:,1]], int))
        d['similarity_scores'].append(_BaseDataset._calculate_box_ious(gg[:,2:6], pp[:,2:6], box_format='xywh'))
    return d


def compact_clear(c, i):
    out = {k:int(c[k]) for k in ['IDSW','CLR_FP','CLR_FN','CLR_TP','Frag']}
    out.update({k:round(float(c[k])*100, 6) for k in ['MOTA','CLR_Re','CLR_Pr']})
    out['IDF1'] = round(float(i['IDF1'])*100, 6)
    return out


def self_test(e):
    g = np.array([[1,1,10,10,5,5,1,1,1],[2,1,10,10,5,5,1,1,1]],float)
    p = np.array([[2,7,10,10,5,5,.9,-1,-1,-1]],float)
    d = official_data(g,p,2)
    c = CLEAR({'THRESHOLD':.5,'PRINT_CONFIG':False}).eval_sequence(d)
    i = Identity({'THRESHOLD':.5,'PRINT_CONFIG':False}).eval_sequence(d)
    assert c['CLR_FN']==1 and c['CLR_FP']==0 and c['IDSW']==0 and c['MOTA']==.5
    assert abs(i['IDF1']-2/3)<1e-12
    gf=defaultdict(list,{1:[(1,10,10,5,5)],2:[(1,10,10,5,5)]})
    pf=defaultdict(list,{2:[(7,10,10,5,5)]})
    acc=e.strict_mot_accumulator(gf,pf,.5)
    result=e.mm.metrics.create().compute(acc, metrics=e.MOT_FIELDS).iloc[0]
    assert result['num_misses']==1 and result['num_switches']==0
    empty=official_data(g,np.zeros((0,10)),2)
    assert CLEAR({'THRESHOLD':.5,'PRINT_CONFIG':False}).eval_sequence(empty)['CLR_FN']==2
    print('PASS: GT-only frame, identity matching, and empty-output scoring',flush=True)


def score(e, gtroot, tracks, names, lengths):
    cm=CLEAR({'THRESHOLD':.5,'PRINT_CONFIG':False});im=Identity({'THRESHOLD':.5,'PRINT_CONFIG':False})
    accum=[];hh={};cc={};ii={}
    for name in names:
        gt=gtroot/name/'gt/gt.txt';pred=tracks/f'{name}.txt'
        gf,pf=e.read_mot_boxes(gt,1),e.read_mot_boxes(pred,-1)
        accum.append(e.strict_mot_accumulator(gf,pf,.5))
        hh[name]=e.HOTA().eval_sequence(e.trackeval_data(gf,pf))
        g=np.loadtxt(gt,delimiter=',',ndmin=2);p=np.loadtxt(pred,delimiter=',',ndmin=2)
        g=g[g[:,6]>=1]
        if not p.size:p=np.zeros((0,10))
        for rows in (g,p):
            assert np.isfinite(rows).all() and np.all(rows[:,4:6]>0)
            assert np.all((rows[:,0]>=1)&(rows[:,0]<=lengths[name]))
            assert len(np.unique(rows[:,:2],axis=0))==len(rows)
        d=official_data(g,p,lengths[name]);cc[name]=cm.eval_sequence(d);ii[name]=im.eval_sequence(d)
    mm=e.mm.metrics.create().compute_many(accum,names=names,metrics=e.MOT_FIELDS,generate_overall=True)
    overall=dict(motmetrics=e.serialise_mot_row(mm.loc['OVERALL']),
                 hota=e.serialise_hota_result(e.HOTA().combine_sequences(hh),e.HOTA()),
                 clear=compact_clear(cm.combine_sequences(cc),im.combine_sequences(ii)))
    per={name:dict(motmetrics=e.serialise_mot_row(mm.loc[name]),
                   hota=e.serialise_hota_result(hh[name],e.HOTA()),clear=compact_clear(cc[name],ii[name])) for name in names}
    assert overall['motmetrics']['num_objects']==37664
    return dict(overall=overall,per_sequence=per)


def run(args,cfg,manifest):
    root=args.root;out=args.output;parent=root/cfg['parent_run'];source=Path(__file__).resolve().parents[2]
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==cfg['code_commit']
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert sha(parent/'outputs.sha256')==cfg['parent_outputs_sha256']
    for line in (parent/'outputs.sha256').read_text().splitlines():
        digest,name=line.split('  ',1);assert sha(parent/name)==digest,name
    pcfg=json.loads((parent/'config.json').read_text());names=pcfg['sequences'];lengths=pcfg['sequence_lengths']
    assert sum(lengths.values())==5754 and len(names)==36
    evalroot=root/cfg['evaluator_source']['path']
    assert subprocess.check_output(['git','-C',str(evalroot),'rev-parse','HEAD'],text=True).strip()==cfg['evaluator_source']['commit']
    assert not subprocess.check_output(['git','-C',str(evalroot),'status','--porcelain'],text=True).strip()
    evpath=evalroot/'tools/evaluate_uavswarm_mot.py';assert sha(evpath)==cfg['evaluator_source']['sha256']
    e=evaluator(evpath);self_test(e)
    refs={x['sequence']:x for x in json.loads((parent/'inference_sequences.json').read_text())}
    replay_checks={};input_hashes={}
    for mode in cfg['modes']:
        tracks=out/'tracks'/mode;tracks.mkdir(parents=True)
        checks={}
        for name in names:
            infofile=root/'datasets/UAVSwarm-dataset-master/test'/name/'seqinfo.ini'
            assert sha(infofile)==pcfg['test_seqinfo_sha256'][name]
            info=sequence_info(infofile.parent);shape=(int(info['imheight']),int(info['imwidth']))
            cache=parent/'detections'/mode/name/'det.txt';input_hashes[str(cache.relative_to(root))]=sha(cache)
            det=cache_detections(cache,lengths[name]);replay=HistoryReplay();rows=[]
            for frame in range(1,lengths[name]+1):
                replay.advance(det[frame],shape,frame)
                for t in replay.tracker.tracked_stracks:
                    if not t.is_activated:continue
                    x,y,w,h=t.feedback_box
                    rows.append(f'{frame},{int(t.track_id)},{x:.1f},{y:.1f},{w:.1f},{h:.1f},{float(t.score):.6f},-1,-1,-1\n')
            path=tracks/f'{name}.txt';path.write_text(''.join(rows))
            digest=sha(path);assert digest==replay.track_hash.hexdigest()
            key={'A':'A_history_tracks_sha256','Mr_closed':'closed_history_tracks_sha256'}.get(mode)
            if key:assert digest==refs[name][key],(mode,name,'parent history mismatch')
            checks[name]=dict(track_sha256=digest,parent_history_match=True if key else None,rows=len(rows))
        replay_checks[mode]=checks;dump(out/'replay_checks.json',replay_checks)
        print('replayed',mode,len(checks),'sequences',flush=True)
    dump(out/'input_cache_sha256.json',input_hashes)
    gtroot=root/pcfg['gt_root']
    for name in names:assert sha(gtroot/name/'gt/gt.txt')==pcfg['canonical_gt_sha256'][name]
    results={}
    for mode in cfg['modes']:
        results[mode]=score(e,gtroot,out/'tracks'/mode,names,lengths)
        dump(out/f'metrics_{mode}.json',results[mode])
        print(mode,json.dumps(results[mode]['overall']),flush=True)
    dump(out/'metrics.json',dict(results=results,sequences=36,frames=5754,gt_boxes=37664,
        tracker=dict(track_thresh=.6,track_buffer=30,match_thresh=.9,mot20=False,frame_rate=30),
        evaluator='strict union-frame motmetrics + TrackEval HOTA; official TrackEval CLEAR/Identity at IoU0.5',
        source_sha256={str(p):sha(p) for p in [evpath,*[Path(inspect.getfile(t)) for t in [CLEAR,Identity,_BaseDataset]]]},
        gt_usage='scoring_only',detector_rerun=False,test_tuning=False))
    manifest.update(status='completed',parent_A_and_closed_replay_match_sequences=72)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();cfg=json.loads(args.config.read_text());out=args.output
    assert not (out/'manifest.yaml').exists()
    import importlib.metadata
    m=dict(status='running',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        private_repository='git@github.com:YUSIO/ByteTrack-MOT.git',branch='exp/194-dense-cva-test',
        code_commit=cfg['code_commit'],code_dirty=False,parent_run=cfg['parent_run'],
        config_sha256=sha(args.config),config=cfg,python=sys.version,host=platform.node(),
        versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','torch','trackeval','motmetrics']},
        device='CPU',detector_rerun=False,training=False,gt_usage='scoring_only',visualization='waived_not_requested')
    dump(out/'manifest.yaml',m);dump(out/'config.json',cfg)
    try:run(args,cfg,m)
    except BaseException as err:
        m.update(status='failed',error=repr(err));raise
    finally:
        m['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();dump(out/'manifest.yaml',m)


if __name__=='__main__':main()
