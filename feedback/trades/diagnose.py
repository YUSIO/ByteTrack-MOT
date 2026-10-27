"""Read-only epoch-one branch audit: no optimizer steps, fixed train/dev probes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import datetime
import numpy as np
import torch
from .model import FrozenTraDeS
from .data import TrainPairs,cva_loss
from .checkpoint import sha256,load_adapter
from .preflight import tensor_hashes


def group(name):
    parts=name.split('.')
    return '.'.join(parts[:2]) if parts[0]=='core' else parts[0]


def rms(value):
    return float(value.detach().float().square().mean().sqrt())


def parameter_changes(model,initial,steps,lr,momentum,decay):
    # Counterfactual SGD movement if gradients are zero tensors, only L2 decay.
    scale=1.;velocity=0.
    for _ in range(steps):
        velocity=momentum*velocity+decay*scale;scale-=lr*velocity
    rows={}
    for name,p in model.adapter.named_parameters():
        old=initial[name].double();new=p.detach().cpu().double();g=group(name)
        item=rows.setdefault(g,dict(parameters=0,initial_sq=0.,final_sq=0.,delta_sq=0.,beyond_decay_sq=0.))
        item['parameters']+=p.numel()
        for key,v in [('initial_sq',old),('final_sq',new),('delta_sq',new-old),('beyond_decay_sq',new-old*scale)]:
            item[key]+=float(v.square().sum())
    for r in rows.values():
        r['relative_change_l2']=(r['delta_sq']/r['initial_sq'])**.5 if r['initial_sq'] else None
        r['relative_beyond_decay_l2']=(r['beyond_decay_sq']/r['initial_sq'])**.5 if r['initial_sq'] else None
        r['final_rms']=(r['final_sq']/r['parameters'])**.5
        r['delta_rms']=(r['delta_sq']/r['parameters'])**.5
    return dict(decay_only_scale=scale,groups=rows)


def correspondence(cva,target):
    q=target['cat_down'].to(cva['h'].device);n=len(q)
    if not n:return dict(pairs=0)
    axes={};errors=[];correct=[]
    for axis in ['h','w']:
        logits=cva[axis][0,q].detach();idx=target[axis+'_ind_down'].to(logits.device)
        prob=(logits*5).softmax(-1);position=(prob*torch.arange(logits.shape[-1],device=logits.device)).sum(-1)
        hits=logits.argmax(-1)==idx;correct.append(hits);errors.append((position-idx)*16)
        axes[axis+'_accuracy']=float(hits.float().mean())
        axes[axis+'_entropy']=float((-(prob*prob.clamp_min(1e-30).log()).sum(-1)).mean())
    return dict(pairs=n,joint_accuracy=float((correct[0]&correct[1]).float().mean()),
                expected_epe_px=float(torch.stack(errors).square().sum(0).sqrt().mean()),**axes)


def gradients(model,det,aux):
    named=list(model.adapter.named_parameters());params=[p for _,p in named]
    a=torch.autograd.grad(det,params,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(.5*aux,params,allow_unused=True)
    groups={}
    for (name,p),x,y in zip(named,a,b):
        item=groups.setdefault(group(name),dict(det_sq=0.,weighted_cva_sq=0.,dot=0.,parameters=0))
        item['parameters']+=p.numel()
        x=None if x is None else x.detach().double();y=None if y is None else y.detach().double()
        if x is not None:item['det_sq']+=float(x.square().sum())
        if y is not None:item['weighted_cva_sq']+=float(y.square().sum())
        if x is not None and y is not None:item['dot']+=float((x*y).sum())
    for r in groups.values():
        r['det_l2']=r['det_sq']**.5;r['weighted_cva_l2']=r['weighted_cva_sq']**.5
        denom=r['det_l2']*r['weighted_cva_l2'];r['cosine']=r['dot']/denom if denom else None
    return groups


def hooks(model,capture,support_zero=False):
    handles=[];core=model.adapter.core
    def record(key):
        def hook(module,args,out):capture[key]=out.detach()
        return hook
    handles.append(core.attention_cur.register_forward_hook(record('attention_cur')))
    handles.append(core.attention_prev.register_forward_hook(record('attention_prev')))
    def dcn_before(module,args):
        support,offset,mask=args
        capture['support']=support.detach();capture['offset']=offset.detach()
        if support_zero:return support*0,offset,mask
    handles.append(core.dcn1_1.register_forward_pre_hook(dcn_before))
    handles.append(core.dcn1_1.register_forward_hook(record('propagated')))
    return handles


def activation_summary(model,raw,heat,residual,capture):
    prop=capture['propagated'];bias=model.adapter.core.dcn1_1.bias.detach()[None,:,None,None]
    weights=torch.cat([capture['attention_cur'],capture['attention_prev']],1).softmax(1)
    offset=capture['offset'].reshape(1,9,2,*heat.shape[-2:])
    h,w=heat.shape[-2:];yy,xx=torch.meshgrid(torch.arange(h,device=heat.device),torch.arange(w,device=heat.device),indexing='ij')
    kernel=torch.tensor([(y-1,x-1) for y in range(3) for x in range(3)],device=heat.device)
    sy=yy[None,None]+kernel[:,0][None,:,None,None]+offset[:,:,0]
    sx=xx[None,None]+kernel[:,1][None,:,None,None]+offset[:,:,1]
    inside=(sy>=0)&(sy<=h-1)&(sx>=0)&(sx<=w-1)
    return dict(raw_p3_rms=rms(raw),residual_rms=rms(residual),relative_residual_rms=rms(residual)/max(rms(raw),1e-30),
        heat_nonzero_fraction=float((heat>0).float().mean()),heat_max=float(heat.max()),heat_mean=float(heat.mean()),
        support_rms=rms(capture['support']),propagated_rms=rms(prop),propagated_without_bias_rms=rms(prop-bias),
        dcn_bias_rms=rms(bias),deform_offset_rms=rms(offset),sampling_inside_fraction=float(inside.float().mean()),
        history_attention_mean=float(weights[:,1].mean()),history_attention_min=float(weights[:,1].min()),
        history_attention_max=float(weights[:,1].max()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data-root','checkpoint','adapter','config','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    config=json.loads(args.config.read_text())
    assert sha256(args.checkpoint)==config['base_sha256']
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==config['gpu_uuid'] and torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write=lambda name,v:(args.output/name).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    manifest=dict(phase='readonly_learning_diagnostic',status='running',optimizer_steps=0,source_run='run_002',
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),test_access=False,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_dirty=False,
        adapter_sha256=sha256(args.adapter),base_sha256=sha256(args.checkpoint),config_sha256=sha256(args.config),
        probe_policy='one midpoint adjacent-frame pair per fit28 and dev6 sequence; GT history; diagnostic only',
        torch=str(torch.__version__),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'])
    from .train import verify_inputs
    manifest['inputs']=verify_inputs(config,args.config,args.data_root,args.checkpoint,'pilot')
    write('manifest.json',manifest)
    model=FrozenTraDeS(args.checkpoint).cuda();initial={k:v.detach().cpu().clone() for k,v in model.adapter.state_dict().items()}
    load_adapter(model,args.adapter,args.checkpoint);trained={k:v.detach().cpu().clone() for k,v in model.adapter.state_dict().items()}
    frozen=tensor_hashes(model.detector);adapter_hash=tensor_hashes(model.adapter)
    write('parameter_changes.json',parameter_changes(model,initial,4625,config['lr'],config['momentum'],config['weight_decay']))
    data=TrainPairs(args.data_root,config['fit_sequences']+config['dev_sequences'])
    indices={pair:i for i,pair in enumerate(data.index)}
    probes=[indices[name,max(2,int(info['seqlength'])//2)] for name,(_,info,_) in data.data.items()]
    records=[]
    for index in probes:
        sample=data[index];name=sample['sequence'];heat=sample['heat'][None].cuda()
        if not heat.max():raise RuntimeError('Empty fixed probe; record rather than silently replace')
        image=sample['image'][None].cuda();previous=sample['previous'][None].cuda();target=sample['targets'][None].cuda()
        model.eval()
        with torch.no_grad():
            features=model.detector.backbone(image);raw=features[0];old=model.detector.backbone(previous)[0]
            model.detector.head.training=True
            off=float(model.detector.head(features,target,image)[0]);model.detector.head.training=False
        record=dict(sequence=name,frame=sample['frame'],partition='fit' if name in config['fit_sequences'] else 'dev',off_detection_loss=off,conditions={})
        reference=None
        for state,mode,variant in [('initial','train','on'),('initial','eval','on'),('trained','train','on'),
                                   ('trained','eval','on'),('trained','eval','support_zero'),
                                   ('trained','eval','heat_shift'),('trained','eval','previous_zero')]:
            model.adapter.load_state_dict(initial if state=='initial' else trained);model.train(mode=='train')
            capture={};handles=hooks(model,capture,variant=='support_zero')
            try:
                refined,cva,residual=model.adapter(raw,old*0 if variant=='previous_zero' else old,
                    heat.roll(heat.shape[-1]//2,-1) if variant=='heat_shift' else heat)
                model.detector.head.training=True
                loss=model.detector.head((refined,*features[1:]),target,image)[0];model.detector.head.training=False
                aux=cva_loss(cva,[sample['cva']])
                stats=dict(detection_loss=float(loss.detach()),cva_loss=float(aux.detach()),
                    correspondence=correspondence(cva,sample['cva']),
                    activations=activation_summary(model,raw,heat,residual,capture))
                if state=='trained' and variant=='on':stats['gradients']=gradients(model,loss,aux)
                if state=='trained' and mode=='eval':
                    if variant=='on':reference=residual.detach().clone()
                    else:
                        stats['residual_change_rms']=rms(residual.detach()-reference)
                        stats['relative_residual_change']=stats['residual_change_rms']/max(rms(reference),1e-30)
                record['conditions'][state+'_'+mode+'_'+variant]=stats
            finally:
                for handle in handles:handle.remove()
            del refined,cva,residual,loss,aux,capture
        records.append(record);write('probes.json',records)
        print(json.dumps(dict(sequence=name,frame=sample['frame'],completed=len(records),total=len(probes))),flush=True)
    model.adapter.load_state_dict(trained);model.eval()
    assert tensor_hashes(model.adapter)==adapter_hash and tensor_hashes(model.detector)==frozen
    assert all(p.grad is None for p in model.parameters())
    manifest.update(status='completed',probe_count=len(records),weights_and_BN_unchanged=True,
        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write('manifest.json',manifest)


if __name__=='__main__':main()
