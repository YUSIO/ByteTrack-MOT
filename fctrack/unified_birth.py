"""A unified, current-frame raw-box birth gate. Original association scores stay unchanged."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def apply_unified_birth(fc, frame, detections, rows, image):
    from fctrack.tracker import iou_matrix
    cfg = fc.unified_birth
    rule = cfg['rule']
    if rule not in ('bridge', 'oracle', 'learned'):
        raise ValueError('unknown unified rule: '+str(rule))
    boxes = np.column_stack((detections[:, :2], detections[:, 2:4]-detections[:, :2])).astype(float)
    taken = np.asarray([row[1] for row in rows if row[1][2]*row[1][3] > 0], float).reshape(-1, 4)
    overlap = iou_matrix(boxes, taken).max(1) if len(taken) else np.zeros(len(boxes))
    consumed = fc.t.used_detection_indices
    eligible = (detections[:, 4] > cfg['score_lower']) & (boxes[:, 2] > 1) & (boxes[:, 3] > 1) & (overlap < cfg['cover_iou'])
    if consumed:
        eligible[list(consumed)] = False
    ids = np.flatnonzero(eligible)
    fc.stats['unified_candidates'] += len(ids)
    q = np.zeros(len(ids), float)
    if rule == 'bridge':
        scores = detections[ids, 4]
        q[scores >= fc.t.det_thresh] = 1.
        need = (scores >= .6) & (scores < fc.t.det_thresh)
        if need.any():
            p, _ = (fc.verify_birth or fc.verify)(image(), boxes[ids[need]])
            q[need] = np.asarray(p)
        accepted = q >= cfg.get('p_min', .97)
    elif rule == 'oracle':
        # Only current GT selects existing raw boxes, never GT coordinates or IDs as output.
        gt = np.asarray(fc.gt.get(frame, np.zeros((0, 4))), float).reshape(-1, 4)
        if len(gt) and len(taken):
            ov = iou_matrix(gt, taken)
            ii, jj = linear_sum_assignment(-(ov*(ov >= .5)))
            gt = np.delete(gt, [i for i,j in zip(ii,jj) if ov[i,j] >= .5], axis=0)
        enabled = np.ones(len(ids), bool) if cfg.get('allow_weak', True) else detections[ids,4] >= .6
        match_ids = np.flatnonzero(enabled)
        if len(gt) and len(match_ids):
            ov = iou_matrix(boxes[ids[match_ids]], gt)
            ii,jj = linear_sum_assignment(-(ov*(ov >= .5)))
            for i,j in zip(ii,jj):
                if ov[i,j] >= .5:
                    q[match_ids[i]] = 1.
        accepted = q > .5
    else:
        if not hasattr(fc, 'unified_predict'):
            raise RuntimeError('learned unified gate has no predictor')
        q = np.asarray(fc.unified_predict(frame, ids, boxes[ids], image), float)
        assert q.shape == (len(ids),) and np.isfinite(q).all()
        accepted = q >= cfg['threshold']
        if not cfg.get('allow_weak', True):
            accepted &= detections[ids,4] >= .6
    if fc.trace is not None:
        fc.trace.record('unified_candidates', detection_indices=ids, boxes=boxes[ids], scores=detections[ids,4], q=q,
                        accepted=accepted, prebirth_rows=[dict(track_id=t, tlwh=b) for t,b,_,_ in rows],
                        used_detection_indices=sorted(consumed), rule=rule)
    for j,k in enumerate(ids):
        fc.unified_query_log.append(dict(frame=frame,detection_index=int(k),score=float(detections[k,4]),
            x=float(boxes[k,0]),y=float(boxes[k,1]),w=float(boxes[k,2]),h=float(boxes[k,3]),cover=float(overlap[k]),q=float(q[j]),accepted=int(accepted[j])))
    added=[]
    order=sorted(np.flatnonzero(accepted),key=lambda j:(-float(detections[ids[j],4]),int(ids[j])))
    for j in order:
        k=int(ids[j]);box=boxes[k].copy();score=detections[k,4]
        if len(taken) and iou_matrix(box,taken).max() >= cfg['cover_iou']:
            fc.stats['unified_final_overlap'] += 1
            if fc.trace is not None:
                fc.trace.record('unified_decision', detection_index=k, decision='final_overlap')
            continue
        same=[t for t in fc.t.tracked_stracks if not t.is_activated and t.frame_id==frame and iou_matrix(box,t.tlwh).max() >= .7]
        if same:
            track=same[0]
            track.re_activate(fc.STrack(box,score),frame,new_id=False)
            fc.stats['unified_reused_unconfirmed'] += 1
        else:
            track=fc.STrack(box,score);track.activate(fc.t.kalman_filter,frame);fc.t.tracked_stracks.append(track)
        track.is_activated=True
        fc.meta[track.track_id]=dict(obs=1,outlier=False,ok=0,cleared=False,real=frame,box=box.copy(),prov=0)
        fc.snap.setdefault(frame,{})[track.track_id]=box[:2]+box[2:]/2
        taken=np.vstack((taken,box));added.append([track.track_id,box.copy(),score,track])
        fc.unified_birth_log.append(dict(frame=frame,detection_index=k,initial_id=int(track.track_id),score=float(score),
             x=float(box[0]),y=float(box[1]),w=float(box[2]),h=float(box[3]),track=track))
        fc.stats['unified_born'] += 1
        fc.stats['weak_born' if score < .6 else 'unified_strong_born'] += 1
        if fc.trace is not None:
            fc.trace.record('unified_decision',detection_index=k,decision='written',final=box,q=float(q[j]),track=fc.trace.track(track,fc.meta))
    return added
