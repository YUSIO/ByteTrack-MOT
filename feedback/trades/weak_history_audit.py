"""Frozen, target-specific history deletion on prespecified dev weak events.

Native detections and history must reproduce the parent byte-for-byte. GT is
used only to locate/score diagnostic queries and assign a previous support
region, never to add support or to change native correspondence or attention.
"""
import argparse
import contextlib
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from feedback.data import read_train_sequences, read_image
from feedback.online import decode
from .alignment_counterfactual import HistoryReplay, cache_detections, detection_lines
from .checkpoint import load_adapter, sha256
from .data import render_heat
from .model import FrozenTraDeS
from .preflight import tensor_hashes
from .vendor.image import gaussian_radius, draw_umich_gaussian


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def ious(box, boxes):
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    if not len(boxes):
        return np.zeros(0)
    a = np.asarray(box, np.float64)
    inter = np.maximum(0, np.minimum(a[:2]+a[2:], boxes[:, :2]+boxes[:, 2:])
                       - np.maximum(a[:2], boxes[:, :2])).prod(1)
    return inter / np.maximum(a[2]*a[3] + boxes[:, 2]*boxes[:, 3] - inter, 1e-30)


def clipped(box, shape):
    b = np.asarray(box, np.float64)
    p = np.maximum(b[:2], 0)
    end = np.minimum(b[:2]+b[2:], [shape[1], shape[0]])
    return np.r_[p, np.maximum(end-p, 0)]


def source_partition(boxes, ratio, size):
    """Partition the actual max-Gaussian at full resolution, then average pool.

    Equal positive maxima split equally. No GT heat is introduced.
    """
    h, w = size
    full = []
    for box in boxes:
        b = np.asarray(box, np.float32).copy()*ratio
        b[[0, 2]] = b[[0, 2]].clip(0, w-1)
        b[[1, 3]] = b[[1, 3]].clip(0, h-1)
        arr = np.zeros((h, w), np.float32)
        bw, bh = b[2:]-b[:2]
        if min(bw, bh) > 0:
            radius = max(0, int(gaussian_radius((math.ceil(bh), math.ceil(bw)))))
            draw_umich_gaussian(arr, ((b[:2]+b[2:])/2).astype(np.int32), radius)
        full.append(torch.from_numpy(arr))
    full = torch.stack(full)
    maximum = full.amax(0)
    winners = (full == maximum[None]) & (maximum[None] > 0)
    parts = maximum[None]*winners/winners.sum(0).clamp_min(1)[None]
    return F.avg_pool2d(parts[:, None], 8, 8)[:, 0], int((winners.sum(0)>1).sum())


def native_capture(adapter, current, previous, heat):
    cap = {'input': [], 'prop': [], 'attention_cur': [], 'attention_prev': []}
    handles = []
    def record(key):
        def hook(module, args, output):
            cap[key].append(output.detach().clone())
        return hook
    def pre(module, args):
        cap['support'], cap['offset'], cap['mask'] = [x.detach().clone() for x in args]
    try:
        for mod, key in [(adapter.input, 'input'), (adapter.core.dcn1_1, 'prop'),
                         (adapter.core.attention_cur, 'attention_cur'),
                         (adapter.core.attention_prev, 'attention_prev')]:
            handles.append(mod.register_forward_hook(record(key)))
        handles.append(adapter.core.dcn1_1.register_forward_pre_hook(pre))
        native, cva, delta = adapter(current, previous, heat)
    finally:
        for h in handles:
            h.remove()
    if cva is None:
        raise ValueError('Prespecified supported event unexpectedly has empty history')
    cap['attention'] = torch.cat(cap['attention_cur']+cap['attention_prev'], 1).softmax(1)[:, 1:2]
    return native, cva, delta, cap


def delete_source(adapter, current, cap, own_heat):
    own_support = cap['input'][1]*own_heat
    bias = adapter.core.dcn1_1.bias[None, :, None, None]
    own_prop = adapter.core.dcn1_1(own_support, cap['offset'], cap['mask'])-bias
    remaining = adapter.core.dcn1_1(cap['support']-own_support, cap['offset'], cap['mask'])
    deleted_delta = adapter.output(cap['attention']*(remaining-bias))
    own_delta = adapter.output(cap['attention']*own_prop)
    return current+deleted_delta, own_prop, own_delta, remaining


def sample(field, points):
    h, w = field.shape[-2:]
    grid = points.clone()
    grid[:, 0] = grid[:, 0]*2/(w-1)-1
    grid[:, 1] = grid[:, 1]*2/(h-1)-1
    return F.grid_sample(field, grid[None, None], align_corners=True,
                         mode='bilinear', padding_mode='zeros')[0, :, 0].T


def box_cover(box, ratio, size, device):
    h, w = [v//8 for v in size]
    b = torch.as_tensor(box, device=device, dtype=torch.float32)*ratio/8
    xx, yy = torch.arange(w, device=device), torch.arange(h, device=device)
    wx = (torch.minimum(xx+1, b[0]+b[2])-torch.maximum(xx, b[0])).clamp(0, 1)
    wy = (torch.minimum(yy+1, b[1]+b[3])-torch.maximum(yy, b[1])).clamp(0, 1)
    return wy[:, None]*wx[None, :]


def score_frame(lines, frame, gt_boxes, shape):
    """COCO's exact per-frame one-to-one matches at fixed score .6."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    anns, identities = [], {}
    for identity, b in gt_boxes.items():
        b = clipped(b, shape)
        if min(b[2:]) <= 0:
            continue
        aid = len(anns)+1; identities[aid] = identity
        anns.append(dict(id=aid, image_id=frame, category_id=1, bbox=b.tolist(),
                         area=float(b[2]*b[3]), iscrowd=0))
    predictions = []
    for line in lines.splitlines():
        r = list(map(float, line.split(',')))
        predictions.append(dict(image_id=frame, category_id=1, bbox=r[2:6], score=r[6]))
    images = [dict(id=frame, width=shape[1], height=shape[0])]
    if not predictions:
        return {k: dict(TP=0, FN=len(anns), FP=0, matched_ids=[]) for k in ('50', '75')}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(); gt.dataset = dict(images=images, annotations=anns, categories=[dict(id=1, name='UAV')]); gt.createIndex()
        dt = gt.loadRes(predictions)
        ev = COCOeval(gt, dt, 'bbox'); ev.evaluate()
    item = next(x for x in ev.evalImgs if x is not None and x['aRng']==ev.params.areaRng[0])
    result = {}
    for label, k in [('50', 0), ('75', 5)]:
        hit = []
        for j, aid in enumerate(item['gtIds']):
            did = int(item['gtMatches'][k, j])
            if did and dt.anns[did]['score'] >= .6:
                hit.append(identities[aid])
        selected = (np.asarray(item['dtScores']) >= .6) & ~item['dtIgnore'][k]
        result[label] = dict(TP=len(hit), FN=len(anns)-len(hit),
                            FP=int(((item['dtMatches'][k]==0)&selected).sum()), matched_ids=hit)
    return result


def dense(prediction, head, ratio):
    raw = decode(prediction.float().cpu(), head.hw, head.strides)[0].numpy().astype(np.float64)
    boxes = np.c_[raw[:, :2]-raw[:, 2:4]/2, raw[:, 2:4]]/ratio
    return raw, boxes


def query_stats(decoded, boxes, query, gt_box):
    obj, cls = decoded[query, 4:6]
    iou = float(ious(gt_box, boxes[query:query+1])[0])
    # One fixed positive-query diagnostic; not dynamic SimOTA or full image loss.
    return dict(query=query, box=boxes[query].tolist(), objectness=float(obj), class_probability=float(cls),
                score=float(obj*cls), iou=iou,
                positive_query_loss_proxy=float(5*(1-iou*iou)-math.log(max(obj, 1e-30))-math.log(max(cls, 1e-30))))


@torch.no_grad()
def run(args, cfg, manifest):
    root = args.root
    if cfg['test_access'] or cfg['optimizer_steps'] != 0:
        raise ValueError('Inference-only protocol')
    for rel, expected in cfg['input_sha256'].items():
        if sha256(root/rel) != expected:
            raise ValueError('Frozen input changed: '+rel)
    manifest['verified_inputs'] = len(cfg['input_sha256'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != cfg['gpu_uuid'] or torch.cuda.device_count()!=1:
        raise ValueError('Wrong physical device')
    torch.set_num_threads(2); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model = FrozenTraDeS(root/cfg['detector'], input_size=cfg['input_size']).cuda().eval()
    metadata = load_adapter(model, root/cfg['adapter'], root/cfg['detector'])
    model.requires_grad_(False)
    if model.adapter.fusion_mode != 'visual_only' or metadata['epoch'] != 10:
        raise ValueError('Wrong checkpoint/fusion')
    before = tensor_hashes(model); head=model.detector.head; head.decode_in_inference=False
    core=model.adapter.core; h,w=model.adapter.feature_size
    cases = json.loads((root/cfg['cases']).read_text())['targets']
    assert len(cases)==20 and len({(r['sequence'],r['frame']) for r in cases})==19
    sequences = list(dict.fromkeys(r['sequence'] for r in cases))
    assert set(sequences)<=set(cfg['allowed_dev_sequences'])
    data=read_train_sequences(root/'datasets/UAVSwarm-dataset-master', sequences)
    all_rows=[]; frame_rows=[]; checks=[]; native_tracks={}
    kernel=torch.tensor([(x-1,y-1) for y in range(3) for x in range(3)],device='cuda',dtype=torch.float32)
    weight=core.dcn1_1.weight.reshape(64,64,9).permute(2,0,1)
    bias=core.dcn1_1.bias[None,:,None,None]
    for seq,(path,info,frames) in data.items():
        wanted={r['frame'] for r in cases if r['sequence']==seq}
        parent=root/cfg['parent_validation']; length=int(info['seqlength'])
        refs=cache_detections(parent/'on'/seq/'det.txt',length)
        lines={mode:{} for mode in ('off','on')}
        for mode in lines:
            for line in (parent/mode/seq/'det.txt').read_text().splitlines():
                f=int(line.split(',')[0]);lines[mode].setdefault(f,[]).append(line+'\n')
        replay=HistoryReplay()
        shape=(int(info['imheight']),int(info['imwidth']))
        for t in range(1,length+1):
            if t in wanted:
                chosen=[r for r in cases if r['sequence']==seq and r['frame']==t]
                image,ratio,shape=read_image(path,t,info,model.input_size)
                previous,old_ratio,old_shape=read_image(path,t-1,info,model.input_size)
                assert ratio==old_ratio and shape==old_shape
                sources=[s for s in replay.tracker.tracked_stracks if s.is_activated and s.score>=.5]
                boxes=np.asarray([s.feedback_box for s in sources],np.float32).reshape(-1,4)
                boxes[:,2:]+=boxes[:,:2]
                assert np.array_equal(boxes,replay.boxes)
                heat=render_heat(boxes*ratio,model.input_size)[None].cuda()
                parts,tie_pixels=source_partition(boxes,ratio,model.input_size);parts=parts.cuda()
                partition_error=float((parts.sum(0)-heat[0,0]).abs().max())
                assert partition_error<1e-6
                features=model.detector.backbone(image[None].cuda())
                old=model.detector.backbone(previous[None].cuda())[0];raw=features[0]
                native,cva,delta,cap=native_capture(model.adapter,raw,old,heat)
                assert (cap['mask']==1).all()
                off_pred=head(features); native_pred=head((native,*features[1:]))
                off_lines=detection_lines(off_pred,head,ratio,t)
                native_lines=detection_lines(native_pred,head,ratio,t)
                parity=dict(sequence=seq,frame=t,off=off_lines==''.join(lines['off'].get(t,[])),
                            native=native_lines==''.join(lines['on'].get(t,[])),heat_partition_error=partition_error)
                checks.append(parity);dump(args.output/'parity.json',checks)
                if not parity['off'] or not parity['native']:
                    raise RuntimeError('Parent output mismatch: '+str(parity))
                dest=args.output/'predictions'/seq/f'{t:06d}';dest.mkdir(parents=True)
                (dest/'off.txt').write_text(off_lines);(dest/'native.txt').write_text(native_lines)
                off_score=score_frame(off_lines,t,frames[t],shape)
                native_score=score_frame(native_lines,t,frames[t],shape)
                off_raw,off_boxes=dense(off_pred,head,ratio)
                native_raw,native_boxes=dense(native_pred,head,ratio)
                frame_rows.append(dict(sequence=seq,frame=t,off=off_score,native=native_score))
                gamma0=core.gamma(torch.zeros_like(cva['offset']),cap['input'][0],cap['input'][1],1,h,w)
                for case in chosen:
                    identity=case['identity'];cb=frames[t][identity];pb=frames[t-1][identity]
                    gt_box=clipped(cb,shape);prev_box=clipped(pb,shape)
                    source_boxes=np.c_[boxes[:,:2],boxes[:,2:]-boxes[:,:2]]
                    overlaps=ious(prev_box,source_boxes);own_indices=np.flatnonzero(overlaps>=.5).tolist()
                    if not own_indices:
                        raise RuntimeError('Prespecified history coverage not reproducible')
                    own_heat=parts[own_indices].sum(0)[None,None]
                    drop,own_prop,own_delta,remaining=delete_source(model.adapter,raw,cap,own_heat)
                    source_error=float((own_prop+remaining-cap['prop'][0]).abs().max())
                    rebuilt=raw+model.adapter.output(cap['attention']*(cap['prop'][0]-bias))
                    native_error=float((rebuilt-native).abs().max())
                    deleted_error=float(((native-drop)-own_delta).abs().max())
                    assert source_error<2e-4 and native_error<2e-5 and deleted_error<2e-4
                    prediction=head((drop,*features[1:]));drop_lines=detection_lines(prediction,head,ratio,t)
                    (dest/f'drop_own_{identity}.txt').write_text(drop_lines)
                    drop_score=score_frame(drop_lines,t,frames[t],shape)
                    drop_raw,drop_boxes=dense(prediction,head,ratio)
                    off_iou=ious(gt_box,off_boxes)
                    # The query is selected once using OFF geometry, never reselected per arm.
                    query=int(np.argmax(off_iou))
                    b=np.asarray(cb,np.float32).copy()*ratio;b[2:]+=b[:2]
                    b[[0,2]]=b[[0,2]].clip(0,model.input_size[1]-1)
                    b[[1,3]]=b[[1,3]].clip(0,model.input_size[0]-1)
                    cc=(b[:2]+b[2:])/2;pc=(pb[:2]+pb[2:]/2)*ratio
                    x,y=(cc/8).astype(int);anchor=raw.new_tensor([x,y]);base9=anchor+kernel
                    qx,qy=x//2,y//2;qi=qy*(w//2)+qx
                    px=(cva['w'][0,qi]*core.tempature).softmax(0)
                    py=(cva['h'][0,qi]*core.tempature).softmax(0)
                    expected=torch.stack(((px*torch.arange(w//2,device='cuda')).sum(),(py*torch.arange(h//2,device='cuda')).sum()))
                    pred_cell=torch.stack((px.argmax(),py.argmax()))
                    gt_cell=raw.new_tensor((pc/16).astype(int))
                    displacement=cva['offset'][0,:,y,x]
                    assert torch.allclose(displacement,2*(expected-raw.new_tensor([qx,qy])),atol=1e-4)
                    desired=anchor+raw.new_tensor((pc-cc)/8)
                    points=base9+cap['offset'].reshape(1,9,2,h,w)[0,:,:,y,x][:,[1,0]]
                    points0=base9+gamma0.reshape(1,9,2,h,w)[0,:,:,y,x][:,[1,0]]
                    sampling={}
                    for label,pts in [('native',points),('zero',base9),('cva_direct',base9+displacement),('gt_translation',base9+(desired-anchor))]:
                        sampled=sample(own_heat,pts)[:,0]
                        sampling[label]=dict(own_mass=float(sampled.sum()),own_hits=int((sampled>1e-8).sum()),
                                             other_mass=float(sample((heat-own_heat).clamp_min(0),pts)[:,0].sum()),
                                             background_hits=int((sample(heat,pts)[:,0]<=1e-8).sum()),
                                             central_error_input_px=float((pts[4]-desired).norm()*8),points_p3=pts.tolist())
                    total_sample=sample(cap['support'],points)
                    reconstruction=torch.einsum('koi,ki->o',weight,total_sample)+core.dcn1_1.bias
                    dcn_error=float((reconstruction-cap['prop'][0][0,:,y,x]).abs().max());assert dcn_error<2e-4
                    cover=box_cover(gt_box,ratio,model.input_size,raw.device)
                    def regional_rms(z):
                        return float(((z[0].square().mean(0)*cover).sum()/cover.sum().clamp_min(1e-20)).sqrt())
                    source_norm=float(own_prop[0,:,y,x].norm());total_norm=float((cap['prop'][0]-bias)[0,:,y,x].norm())
                    cva_motion=displacement*8-raw.new_tensor(pc-cc)
                    prev_diag=float(np.linalg.norm(prev_box[2:]))
                    q_stats={name:query_stats(a,bs,query,gt_box) for name,a,bs in [('off',off_raw,off_boxes),('native',native_raw,native_boxes),('drop_own',drop_raw,drop_boxes)]}
                    target_caches={}
                    for name,text in [('off',off_lines),('native',native_lines),('drop_own',drop_lines)]:
                        parsed=np.array([list(map(float,line.split(','))) for line in text.splitlines()])
                        overlaps_now=ious(gt_box,parsed[:,2:6]);valid=overlaps_now>=.5
                        target_caches[name]=dict(max_iou=float(overlaps_now.max()),
                            best_score_iou50=float(parsed[valid,6].max()) if valid.any() else None)
                    collisions=[]
                    for other,ob in frames[t].items():
                        bb=np.asarray(ob,np.float32).copy()*ratio;bb[2:]+=bb[:2]
                        bb[[0,2]]=bb[[0,2]].clip(0,model.input_size[1]-1);bb[[1,3]]=bb[[1,3]].clip(0,model.input_size[0]-1)
                        qq=((bb[:2]+bb[2:])/2/16).astype(int)
                        if np.array_equal(qq,[qx,qy]):collisions.append(int(other))
                    row=dict(**case,previous_frame=t-1,ratio=ratio,current_box=cb.tolist(),previous_box=pb.tolist(),
                        source_track_ids=[int(sources[i].track_id) for i in own_indices],source_ious=overlaps[own_indices].tolist(),
                        source_ambiguous=len(own_indices)!=1,heat_tie_pixels=tie_pixels,p3_query=[x,y],cva_query=[qx,qy],
                        own_heat_peak=float(own_heat.max()),history_attention_at_query=float(cap['attention'][0,0,y,x]),
                        same_cell_gt_ids=collisions,cva_argmax_cell=pred_cell.tolist(),cva_expected_cell=expected.tolist(),gt_previous_cell=gt_cell.tolist(),
                        cva_argmax_correct=bool((pred_cell==gt_cell).all()),zero_motion_cell_correct=bool((raw.new_tensor([qx,qy])==gt_cell).all()),
                        cva_quantized_epe_input_px=float((expected-gt_cell).norm()*16),
                        cva_motion_error_original_px=float(cva_motion.norm()/ratio),
                        cva_motion_error_target_diagonals=float(cva_motion.norm()/ratio/max(prev_diag,1e-20)),
                        gamma_response_to_cva_input_px=float((points-points0).square().sum(1).mean().sqrt()*8),
                        sampling=sampling,own_propagated_query_norm=source_norm,total_propagated_query_norm=total_norm,
                        own_p3_roi_rms=regional_rms(own_delta),all_history_p3_roi_rms=regional_rms(delta),raw_p3_roi_rms=regional_rms(raw),
                        fixed_off_query=q_stats,fixed_query_level=next(i for i,end in enumerate(np.cumsum([hh*ww for hh,ww in head.hw])) if query<end),
                        cached_target=target_caches,frame_detection=dict(off=off_score,native=native_score,drop_own=drop_score),
                        audits=dict(source_error=source_error,native_error=native_error,deleted_error=deleted_error,dcn_error=dcn_error))
                    all_rows.append(row)
                    # Compact diagnostics only: no full image feature dump.
                    np.savez_compressed(args.output/f'{seq}_{t:06d}_{identity}_diagnostic.npz',
                        px=px.cpu().numpy(),py=py.cpu().numpy(),own_heat=own_heat[0,0].cpu().numpy(),
                        own_delta_rms=own_delta[0].square().mean(0).sqrt().cpu().numpy())
                    dump(args.output/'targets.json',all_rows)
                    print(json.dumps(dict(sequence=seq,frame=t,identity=identity,cva_correct=row['cva_argmax_correct'],
                        own_hits=sampling['native']['own_hits'],own_mass=sampling['native']['own_mass'],
                        score_native=q_stats['native']['score'],score_drop=q_stats['drop_own']['score'])),flush=True)
            replay.advance(refs[t],shape,t)
        expected_track=sha256(parent/'on'/seq/'tracks.txt')
        native_tracks[seq]=replay.track_hash.hexdigest()==expected_track
        assert native_tracks[seq]
    assert len(all_rows)==20 and len(frame_rows)==19
    if tensor_hashes(model)!=before or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Weights/BN/gradients changed')
    dump(args.output/'frames.json',frame_rows)
    dump(args.output/'metrics.json',dict(targets=20,frame_pairs=19,
        cva_correct=sum(r['cva_argmax_correct'] for r in all_rows),
        zero_motion_correct=sum(r['zero_motion_cell_correct'] for r in all_rows),
        native_tracks_match=native_tracks,all_weights_and_BN_unchanged=True,
        optimizer_steps=0,test_access=False,scope='target-specific fixed-state diagnostic, not a deployed tracker'))
    manifest.update(status='completed',all_weights_and_BN_unchanged=True,native_tracks_match=native_tracks,
                    target_count=20,frame_pairs=19,gpu_name=torch.cuda.get_device_name(0))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('root','config','output'):ap.add_argument('--'+key,type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'manifest.yaml').exists():raise FileExistsError('Run already exists')
    dirty=subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    if dirty:raise RuntimeError('Clean source required')
    cfg=json.loads(args.config.read_text())
    manifest=dict(status='running',phase='frozen_weak_history_target_audit',training=False,optimizer_steps=0,test_access=False,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),effective_argv=sys.argv,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_dirty=False,
        config_sha256=sha256(args.config),python_executable=sys.executable,python=sys.version,
        torch=str(torch.__version__),torchvision=str(torchvision.__version__),numpy=np.__version__,platform=platform.platform(),
        private_repository='git@github.com:YUSIO/ByteTrack-MOT.git',branch='exp/194-weak-history-audit',
        base_commit='659fa0a43d5326f85c19ddac5507780f8fef4396',gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'))
    dump(args.output/'manifest.yaml',manifest);dump(args.output/'config.json',cfg)
    try:run(args,cfg,manifest)
    except BaseException as error:manifest.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        manifest['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();dump(args.output/'manifest.yaml',manifest)


if __name__=='__main__':main()
