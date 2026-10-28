"""Six-pair frozen-attention CVA/gamma counterfactual; no training or tracking."""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch
import torchvision
from torch.nn import functional as F
from .checkpoint import load_adapter, sha256
from .data import TrainPairs
from .model import FrozenTraDeS
from .preflight import tensor_hashes
from .vendor.image import draw_umich_gaussian, gaussian_radius

MODES = ('A_native', 'B_no_neighbors', 'C_identity_gain', 'D_center_anchor',
         'R_only', 'd_only', 'GT_local', 'R_GT_local', 'OFF')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def split_offsets(core, d, native):
    """Return exact affine terms in native [k, y/x] DCN channel order."""
    h = F.conv2d(d[:, 1:2], core.conv_offset_h.weight[:, :1], padding=1)
    w = F.conv2d(d[:, :1], core.conv_offset_w.weight[:, :1], padding=1)
    cv = torch.stack((h, w), 2).flatten(1, 2)
    sh = core.conv_offset_h.weight[:, 0].sum((-1, -2))[None, :, None, None] * d[:, 1:2]
    sw = core.conv_offset_w.weight[:, 0].sum((-1, -2))[None, :, None, None] * d[:, :1]
    sd = torch.stack((sh, sw), 2).flatten(1, 2)
    direct = d[:, [1, 0]][:, None].expand(-1, 9, -1, -1, -1).flatten(1, 2)
    remainder = native - cv
    neighbor = cv - sd
    return dict(A_native=native, B_no_neighbors=remainder + sd,
                C_identity_gain=remainder + direct + neighbor,
                D_center_anchor=remainder + direct, R_only=remainder,
                d_only=direct), remainder


def source_parts(gt, ratio, device):
    ids, full = sorted(gt), []
    for identity in ids:
        b = gt[identity].copy() * ratio
        b[2:] += b[:2]
        b[[0, 2]] = b[[0, 2]].clip(0, 1439)
        b[[1, 3]] = b[[1, 3]].clip(0, 799)
        a = np.zeros((800, 1440), np.float32)
        w, h = b[2:] - b[:2]
        if min(w, h) > 0:
            draw_umich_gaussian(a, ((b[:2] + b[2:])/2).astype(np.int32),
                               max(0, int(gaussian_radius((math.ceil(h), math.ceil(w))))))
        full.append(torch.from_numpy(a))
    full = torch.stack(full).to(device)
    maximum = full.amax(0)
    winners = (full == maximum[None]) & (maximum[None] > 0)
    parts = maximum[None] * winners / winners.sum(0).clamp_min(1)[None]
    return ids, F.avg_pool2d(parts[:, None], 8, 8)[:, 0]


def box_cover(box, ratio, device):
    b = torch.tensor(box, device=device) * ratio / 8
    xx, yy = torch.arange(180, device=device), torch.arange(100, device=device)
    wx = (torch.minimum(xx+1, b[0]+b[2])-torch.maximum(xx, b[0])).clamp(0, 1)
    wy = (torch.minimum(yy+1, b[1]+b[3])-torch.maximum(yy, b[1])).clamp(0, 1)
    return wy[:, None] * wx[None, :]


def gt_local_offsets(spec, native, remainder):
    """Oracle only in matched GT-box-intersecting cells; native outside.

    Overlap owner: smallest normalized center distance, then smallest ID.
    This is a partial/local oracle, not an oracle full-image motion field.
    """
    device = native.device
    yy, xx = torch.meshgrid(torch.arange(100, device=device)+.5,
                           torch.arange(180, device=device)+.5, indexing='ij')
    best = torch.full((100, 180), float('inf'), device=device)
    field = torch.zeros((2, 100, 180), device=device)
    for o in sorted(spec['objects'], key=lambda x:x['identity']):
        cb = torch.tensor(o['current_box'], device=device) * spec['ratio'] / 8
        pb = torch.tensor(o['previous_box'], device=device) * spec['ratio'] / 8
        cc, pc = cb[:2]+cb[2:]/2, pb[:2]+pb[2:]/2
        score = ((xx-cc[0])/cb[2].clamp_min(1e-6))**2 + ((yy-cc[1])/cb[3].clamp_min(1e-6))**2
        choose = (box_cover(o['current_box'], spec['ratio'], device)>0) & (score<best)
        best[choose] = score[choose]
        field[:, choose] = (pc-cc)[:, None]
    valid = torch.isfinite(best)
    direct = field[[1, 0]][None, None].expand(1, 9, -1, -1, -1).flatten(1, 2)
    return torch.where(valid[None, None], direct, native), torch.where(valid[None, None], remainder+direct, native), valid


def sample(value, points):
    grid = points.clone()
    grid[:, 0] = grid[:, 0]*2/179-1
    grid[:, 1] = grid[:, 1]*2/99-1
    return F.grid_sample(value, grid[None, None], align_corners=True)[0, :, 0].T


def losses(model, features, target, image):
    head = model.detector.head
    assert not any(m.training for m in head.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))
    try:
        head.training = True  # dispatch loss only; every child BN remains eval
        values = head(features, target, image)
    finally:
        head.training = False
    result = {k:float(v) for k,v in zip(('total_loss','iou_loss','conf_loss','cls_loss','l1_loss','num_fg'), values)}
    assert all(math.isfinite(v) for v in result.values())
    return result


@torch.no_grad()
def run(args, cfg, manifest):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == cfg['gpu_uuid']
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert cfg['modes'] == list(MODES)
    inputs = cfg['input_sha256']
    for path, digest in inputs.items():
        assert sha256(args.root/path) == digest, path
    assert sha256(args.root/cfg['adapter']) == cfg['adapter_sha256']
    assert sha256(args.root/cfg['base']) == cfg['base_sha256']
    specs = json.loads((args.root/cfg['specs']).read_text())
    assert len(specs) == 6 and sum(len(r['objects']) for r in specs) == 51
    assert [r['sequence'] for r in specs] == cfg['sequences']
    model = FrozenTraDeS(args.root/cfg['base']).cuda().eval().requires_grad_(False)
    metadata = load_adapter(model, args.root/cfg['adapter'], args.root/cfg['base'])
    assert metadata['epoch'] == 10 and model.adapter.fusion_mode == 'visual_only'
    before = tensor_hashes(model)
    core = model.adapter.core
    # Adapter.forward normally sets this non-buffer device before calling CVA.
    # This diagnostic calls the core directly to capture native intermediates.
    core.opt.device = next(model.parameters()).device
    bias = core.dcn1_1.bias[None, :, None, None]
    weight = core.dcn1_1.weight.reshape(64,64,9).permute(2,0,1)
    kernel = torch.tensor([(x-1,y-1) for y in range(3) for x in range(3)], device='cuda')
    data = TrainPairs(args.root/'datasets/UAVSwarm-dataset-master', cfg['sequences'])
    indices = {x:i for i,x in enumerate(data.index)}
    frames, objects, audits = [], [], []
    for spec in specs:
        name, frame = spec['sequence'], spec['frame']
        s = data[indices[name,frame]]
        image, previous, target, heat = [s[k][None].cuda() for k in ('image','previous','targets','heat')]
        features = model.detector.backbone(image)
        raw = features[0]
        old = model.detector.backbone(previous)[0]
        cur, prev = model.adapter.input(raw), model.adapter.input(old)
        support = prev * heat
        captured = {}
        def pre_hook(module, values):
            captured['offset'], captured['mask'] = values[1].clone(), values[2].clone()
        def post_hook(module, values, output): captured['prop'] = output.clone()
        def att_hook(module, values, output): captured['att_logits'] = output.clone()
        handles = [core.dcn1_1.register_forward_pre_hook(pre_hook), core.dcn1_1.register_forward_hook(post_hook),
                   core.attention_prev.register_forward_hook(att_hook)]
        result = core.CVA_MFW([cur,prev], [support], visual_only=True)
        for h in handles: h.remove()
        attention = torch.cat((core.attention_cur(cur),captured['att_logits']),1).softmax(1)[:,1:2].clone()
        attention_digest = tensor_sha(attention)
        d, native = result[2], captured['offset']
        offsets, remainder = split_offsets(core, d, native)
        oracle, r_oracle, oracle_mask = gt_local_offsets(spec, native, remainder)
        offsets.update(GT_local=oracle, R_GT_local=r_oracle)
        zero_ref = core.gamma(torch.zeros_like(d),cur,prev,1,100,180)
        algebra = float((zero_ref-remainder).abs().max())
        assert algebra < cfg['tolerances']['offset_abs']
        native_residual = model.adapter.output(attention*(captured['prop']-bias))
        native_error = float((native_residual-model.adapter.output(result[0][0])).abs().max())
        assert native_error < cfg['tolerances']['native_residual_abs']
        ids, parts = source_parts(data.data[name][2][frame-1],spec['ratio'],raw.device)
        assert float((parts.sum(0)-heat[0,0]).abs().max()) < 1e-6
        maps = {'attention':attention[0,0].cpu().numpy(), 'cva':d[0].cpu().numpy(),
                'oracle_mask':oracle_mask.cpu().numpy()}
        for mode in MODES:
            if mode == 'OFF':
                residual = raw*0
                loss = losses(model,features,target,image)
                frames.append(dict(sequence=name,frame=frame,mode=mode,attention_sha256=attention_digest,
                    residual_relative_rms=0.,**loss))
                continue
            off = offsets[mode]
            prop = captured['prop'] if mode=='A_native' else core.dcn1_1(support,off,captured['mask'])
            visual = prop-bias
            residual = model.adapter.output(attention*visual)
            loss = losses(model,(raw+residual,*features[1:]),target,image)
            frames.append(dict(sequence=name,frame=frame,mode=mode,attention_sha256=attention_digest,
                residual_relative_rms=float(residual.square().mean().sqrt()/raw.square().mean().sqrt()),**loss))
            # Zero support with this same frozen attention must give exact bypass.
            zero = core.dcn1_1(torch.zeros_like(support),off,captured['mask'])-bias
            assert float(zero.abs().max()) <= cfg['tolerances']['zero_support_abs']
            maps[mode+'_residual_rms'] = residual[0].square().mean(0).sqrt().cpu().numpy()
            maps[mode+'_offset'] = off[0].cpu().numpy()
            for obj in spec['objects']:
                identity = obj['identity']
                actual_box = data.data[name][2][frame][identity]
                assert np.array_equal(actual_box,np.array(obj['current_box']))
                assert np.array_equal(data.data[name][2][frame-1][identity],np.array(obj['previous_box']))
                cb = torch.tensor(obj['current_box'],device=raw.device)*spec['ratio']
                pb = torch.tensor(obj['previous_box'],device=raw.device)*spec['ratio']
                cc, pc = cb[:2]+cb[2:]/2,pb[:2]+pb[2:]/2
                q = (cc/8).floor()
                x,y = [int(v) for v in q]
                pts = q+kernel+off.reshape(1,9,2,100,180)[0,:,:,y,x][:,[1,0]]
                ideal = q+kernel+(pc-cc)/8
                ownheat = parts[ids.index(identity)]
                ownsupport = prev*ownheat[None,None]
                ownprop = core.dcn1_1(ownsupport,off,captured['mask'])-bias
                own_samples, all_samples = sample(ownsupport,pts),sample(support,pts)
                own_contrib = torch.einsum('koi,ki->ko',weight,own_samples)
                all_contrib = torch.einsum('koi,ki->ko',weight,all_samples)
                recon = float((all_contrib.sum(0)-visual[0,:,y,x]).abs().max())
                source_recon = float((own_contrib.sum(0)-ownprop[0,:,y,x]).abs().max())
                assert max(recon,source_recon)<cfg['tolerances']['dcn_reconstruction_abs']
                cover = box_cover(obj['current_box'],spec['ratio'],raw.device)
                energy = ownprop[0].square().mean(0)
                den = energy.sum().clamp_min(1e-30)
                yy,xx = torch.meshgrid(torch.arange(100,device=raw.device)+.5,torch.arange(180,device=raw.device)+.5,indexing='ij')
                centroid = torch.stack(((energy*xx).sum(),(energy*yy).sum()))/den*8
                ownmass, totalmass = sample(ownheat[None,None],pts)[:,0],sample(heat,pts)[:,0]
                other = visual[0,:,y,x]-ownprop[0,:,y,x]
                mean_error = ((pts-ideal).square().sum(1).mean().sqrt()*8)
                objects.append(dict(sequence=name,frame=frame,identity=identity,mode=mode,
                    current_box=obj['current_box'],previous_box=obj['previous_box'],ratio=spec['ratio'],
                    points_p3=pts.cpu().tolist(),cva_displacement_p3=d[0,:,y,x].cpu().tolist(),
                    gt_displacement_p3=((pc-cc)/8).cpu().tolist(),centre_error_px=float((pts[4]-ideal[4]).norm()*8),
                    lattice_error_rms_px=float(mean_error), own_support_hits=int((ownmass>1e-8).sum()),
                    own_heat_sum=float(ownmass.sum()),other_heat_sum=float((totalmass-ownmass).sum()),
                    own_box_energy_fraction=float((energy*cover).sum()/den),
                    own_energy_centroid_error_px=float((centroid-cc).norm()),
                    own_query_norm=float(ownprop[0,:,y,x].norm()),other_query_norm=float(other.norm()),
                    own_kernel_rms=own_contrib.square().mean(1).sqrt().cpu().tolist(),
                    other_kernel_rms=(all_contrib-own_contrib).square().mean(1).sqrt().cpu().tolist(),
                    history_to_current_norm=float(visual[0,:,y,x].norm()/cur[0,:,y,x].norm().clamp_min(1e-30)),
                    dcn_reconstruction_abs=recon,source_reconstruction_abs=source_recon))
                if [name,identity] in cfg['figure_cases']:
                    maps[f'{mode}_own_{identity}'] = energy.sqrt().cpu().numpy()
            assert tensor_sha(attention)==attention_digest
            print(json.dumps(dict(sequence=name,frame=frame,mode=mode,loss=loss['total_loss'])),flush=True)
        np.savez_compressed(args.output/f'{name}_maps.npz',**maps)
        audits.append(dict(sequence=name,frame=frame,offset_decomposition_abs=algebra,
            native_residual_reconstruction_abs=native_error,attention_sha256=attention_digest,
            oracle_p3_cells=int(oracle_mask.sum()),matched_objects=len(spec['objects']),
            detection_loss_gt_objects=int((target[0,:,3]>0).sum())))
        dump(args.output/'per_frame.json',frames)
        dump(args.output/'objects.json',objects)
        dump(args.output/'audits.json',audits)
    after = tensor_hashes(model)
    assert before == after and all(p.grad is None for p in model.parameters())
    for path,digest in inputs.items(): assert sha256(args.root/path)==digest
    assert sha256(args.root/cfg['adapter'])==cfg['adapter_sha256']
    with (args.output/'per_frame.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(frames[0]));writer.writeheader();writer.writerows(frames)
    means={}
    for mode in MODES:
        ff=[r for r in frames if r['mode']==mode]
        oo=[r for r in objects if r['mode']==mode]
        means[mode]=dict(loss={k:float(np.mean([r[k] for r in ff])) for k in ('total_loss','iou_loss','conf_loss','cls_loss','num_fg')})
        if oo:
            means[mode]['geometry']={k:float(np.mean([r[k] for r in oo])) for k in ('centre_error_px','lattice_error_rms_px','own_support_hits','own_heat_sum','own_box_energy_fraction','own_energy_centroid_error_px','own_query_norm','other_query_norm','history_to_current_norm')}
    dump(args.output/'metrics.json',dict(means=means,frames=6,objects=51,weights_and_BN_unchanged=True,
        attention_frozen=True,optimizer_steps=0,test_access=False,oracle_scope=cfg['oracle_scope']))
    dump(args.output/'state_hashes.json',before)
    manifest.update(status='completed',weights_and_BN_unchanged=True,attention_frozen=True,frames=6,objects=51)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','config','output'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'manifest.yaml').exists():raise FileExistsError('Attempt already exists')
    cfg=json.loads(args.config.read_text())
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert revision==cfg['code_commit']
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    manifest=dict(status='running',phase='six_pair_frozen_gamma_counterfactual',code_commit=revision,
        source_dirty=False,parent_run='run_001/epoch10',config_sha256=sha256(args.config),
        python_executable=sys.executable,python=sys.version,torch=str(torch.__version__),torchvision=str(torchvision.__version__),
        platform=platform.platform(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),effective_argv=sys.argv,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),optimizer_steps=0,test_access=False)
    dump(args.output/'manifest.yaml',manifest)
    dump(args.output/'config.json',cfg)
    try:run(args,cfg,manifest)
    except BaseException as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        manifest['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        dump(args.output/'manifest.yaml',manifest)


if __name__=='__main__':main()
