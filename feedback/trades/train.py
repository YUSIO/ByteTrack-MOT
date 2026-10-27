"""Immutable single-epoch frozen-detector pilot and independent CUDA smoke."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from feedback.data import read_image
from .checkpoint import sha256,save_adapter
from .data import TrainPairs,cva_loss
from .model import FrozenTraDeS
from .online import TraDeSOnline
from .preflight import tensor_hashes


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def check_config(config):
    fit,dev=config['fit_sequences'],config['dev_sequences']
    if not fit or not dev or len(fit)!=len(set(fit)) or len(dev)!=len(set(dev)) or set(fit)&set(dev):
        raise ValueError('Nonempty unique disjoint fit/dev required')
    if any(not n.startswith('UAVSwarm-') or Path(n).name!=n for n in fit+dev):
        raise ValueError('Invalid sequence name')
    if set(config['quarantine_sequences'])&set(fit+dev):raise ValueError('Quarantine overlap')
    if config['epochs']!=1 or config['batch_size']!=1 or config['precision']!='fp32':
        raise ValueError('This pilot implements exactly one epoch, batch one, FP32')
    if config['test_access'] or config['resume'] or config['gradient_clip'] is not None:
        raise ValueError('No test, resume, or clipping in pilot v1')


def verify_inputs(config,config_path,root,checkpoint,phase):
    if sha256(checkpoint)!=config['base_sha256']:raise ValueError('Wrong base detector')
    count=0;records={}
    for spec in config['inventories']:
        path=config_path.parent/spec['file']
        if sha256(path)!=spec['sha256']:raise ValueError('Inventory changed')
        seen=set()
        for line in path.read_text().splitlines():
            expected,relative=line.split('  ',1);p=Path(relative)
            if p.is_absolute() or '..' in p.parts or p.parts[0]!='train' or p.parts[1] not in config['fit_sequences']+config['dev_sequences']:
                raise ValueError('Inventory outside train fit/dev')
            seen.add(relative)
            if phase=='smoke' and p.parts[1]!='UAVSwarm-05':continue
            if sha256(root/p)!=expected:raise ValueError(f'Input changed: {relative}')
            count+=1
        # Inventories must cover all images and annotations, not merely a subset.
        for name in config[spec['sequences_key']]:
            from feedback.data import sequence_info
            info=sequence_info(root/'train'/name)
            required={f'train/{name}/gt/gt.txt',f'train/{name}/seqinfo.ini'}
            required.update(f"train/{name}/{info.get('imdir','img1')}/{f:06d}{info.get('imext','.jpg')}"
                            for f in range(1,int(info['seqlength'])+1))
            if not required<=seen:raise ValueError('Incomplete inventory')
        records[spec['file']]=spec['sha256']
    return dict(base_sha256=config['base_sha256'],inventory_sha256=records,verified_files=count)


def check_gradients(model):
    if any(p.grad is not None for p in model.detector.parameters()):raise RuntimeError('Detector gradient')
    for name,p in model.adapter.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError(f'Nonfinite gradient: {name}')
    return float(sum(p.grad.detach().double().square().sum() for p in model.adapter.parameters() if p.grad is not None).sqrt())


def run(args,config,manifest):
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Exactly one visible CUDA device required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=config['gpu_uuid']:raise RuntimeError('Wrong physical GPU binding')
    torch.set_num_threads(2);random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed']);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest['inputs']=verify_inputs(config,args.config,args.data_root,args.checkpoint,args.phase)
    manifest['device_name']=torch.cuda.get_device_name(0)
    write_json(args.output/'manifest.yaml',manifest)
    device=torch.device('cuda:0')
    model=FrozenTraDeS(args.checkpoint,input_size=config['input_size']).to(device)
    frozen=tensor_hashes(model.detector);write_json(args.output/'detector_initial_sha256.json',frozen)
    samples=TrainPairs(args.data_root,['UAVSwarm-05'] if args.phase=='smoke' else config['fit_sequences'],config['input_size'])
    order=[0,1] if args.phase=='smoke' else np.random.default_rng(config['seed']).permutation(len(samples)).tolist()
    if args.phase=='pilot' and len(samples)!=config['expected_fit_pairs']:raise ValueError('Fit frame count changed')
    write_json(args.output/'sample_order.json',[dict(index=i,sequence=samples.index[i][0],frame=samples.index[i][1]) for i in order])
    optimizer=torch.optim.SGD(model.optimizer_parameters(),lr=config['lr'],momentum=config['momentum'],weight_decay=config['weight_decay'])
    if {id(p) for group in optimizer.param_groups for p in group['params']}!={id(p) for p in model.adapter.parameters()}:
        raise RuntimeError('Optimizer ownership')
    # Same CUDA execution path: raw OFF and initial ON must match bitwise.
    example=samples[0]
    x=example['image'][None].to(device);old=example['previous'][None].to(device);heat=example['heat'][None].to(device)
    model.eval()
    with torch.no_grad():
        original=model(x,enabled=False)['output'].clone()
        initial=model(x,previous_images=old,heat=heat)['output']
        if not torch.equal(original,initial):raise RuntimeError('Nonidentity initialization')
    del initial
    writer=SummaryWriter(str(args.output/'tensorboard'))
    started=time.monotonic();updates=skipped=0
    try:
        with (args.output/'training.csv').open('w',buffering=1) as log:
            fields=['step','sequence','frame','detection_loss','cva_loss','total_loss','grad_norm','pairs','collisions','skipped','elapsed_s']
            csvlog=csv.DictWriter(log,fieldnames=fields);csvlog.writeheader()
            for step,index in enumerate(order,1):
                sample=samples[index];model.train();optimizer.zero_grad(set_to_none=True)
                entry=dict(step=step,sequence=sample['sequence'],frame=sample['frame'],pairs=sample['pair_stats']['used_pairs'],
                    collisions=sample['pair_stats']['collision_excluded_pairs'],skipped=0)
                if sample['heat'].max()==0:
                    skipped+=1;entry.update(detection_loss=0,cva_loss=0,total_loss=0,grad_norm=0,skipped=1)
                else:
                    result=model(sample['image'][None].to(device),previous_images=sample['previous'][None].to(device),
                        heat=sample['heat'][None].to(device),targets=sample['targets'][None].to(device))
                    det=result['output']['total_loss'];aux=cva_loss(result['cva'],[sample['cva']]);loss=det+config['cva_weight']*aux
                    if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at step {step}')
                    loss.backward();grad=check_gradients(model);optimizer.step();updates+=1
                    entry.update(detection_loss=float(det.detach()),cva_loss=float(aux.detach()),total_loss=float(loss.detach()),grad_norm=grad)
                    del result,det,aux,loss
                entry['elapsed_s']=time.monotonic()-started;csvlog.writerow(entry)
                for key in ['detection_loss','cva_loss','total_loss','grad_norm']:writer.add_scalar('train/'+key,entry[key],step)
                if step<=2 or step%25==0 or step==len(order):
                    print(json.dumps(dict(phase='train',total_steps=len(order),**entry)),flush=True);writer.flush()
                    write_json(args.output/'progress.json',dict(phase='train',total_steps=len(order),**entry))
            writer.flush()
        if tensor_hashes(model.detector)!=frozen:raise RuntimeError('Detector parameters or BN changed')
        model.eval()
        with torch.no_grad():
            off=model(x,enabled=False)['output'];empty=model(x,previous_images=old,heat=heat*0)['output']
            on=model(x,previous_images=old,heat=heat)['output']
            if not torch.equal(original,off) or not torch.equal(original,empty):raise RuntimeError('Frozen baseline parity failed')
            if not torch.isfinite(on).all():raise FloatingPointError('Nonfinite post-training prediction')
            delta=float((on-original).abs().max())
        metrics=dict(phase=args.phase,updates=updates,skipped=skipped,detector_frozen_tensors=len(frozen),
            off_empty_bitwise_identical=True,on_max_raw_delta=delta,training_seconds=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),performance_claim=False)
        save_adapter(model,args.output/'last.pt',args.checkpoint,dict(phase=args.phase,seed=config['seed'],epoch=1,diagnostic_only=args.phase=='smoke'))
        torch.save(dict(optimizer=optimizer.state_dict(),seed=config['seed'],epoch=1,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),args.output/'optimizer_last.pt')
        if args.phase=='smoke':
            if delta<=0:raise RuntimeError('No trained feedback effect')
            runner=TraDeSOnline(model);runner.reset('UAVSwarm-05');trace=[]
            path,info,_=samples.data['UAVSwarm-05']
            for frame in (1,2,3):
                image,ratio,shape=read_image(path,frame,info,config['input_size'])
                result=runner.step(image[None].to(device),'UAVSwarm-05',frame,shape,ratio)
                trace.append(dict(frame=frame,feedback_used=result['feedback_used']))
            if trace[0]['feedback_used'] or not any(r['feedback_used'] for r in trace[1:]):raise RuntimeError('Causal ON disconnected')
            metrics['online']=trace
        else:
            from .validate import validate
            validation={}
            for mode in config['validation_modes']:
                write_json(args.output/'progress.json',dict(phase='validation',mode=mode))
                validation[mode]=validate(model,args.data_root,config['dev_sequences'],args.output/'validation'/mode,mode,device)
                for key,value in validation[mode]['metrics'].items():writer.add_scalar('dev/'+mode+'/'+key,value,1)
            on,off,shift=[validation[k] for k in ['on','off','heat_shift']]
            checks=dict(ap_gt_off=on['metrics']['AP50:95']>off['metrics']['AP50:95'],
                ap_gt_heat_shift=on['metrics']['AP50:95']>shift['metrics']['AP50:95'],
                ap75_ge_off=on['metrics']['AP75']>=off['metrics']['AP75'],
                fp_le_off=on['fixed_iou50_score06']['FP']<=off['fixed_iou50_score06']['FP'])
            metrics.update(validation=validation,gate_checks=checks,pilot_gate_passed=all(checks.values()),
                selected_epoch=1,selection='sole_epoch_by_preregistered_on_dev_AP50:95; candidate_not_deployment_approval')
            shutil.copyfile(args.output/'last.pt',args.output/'best.pt')
        if tensor_hashes(model.detector)!=frozen:raise RuntimeError('Detector changed during validation')
        write_json(args.output/'metrics.json',metrics)
        print(json.dumps({k:v for k,v in metrics.items() if k!='validation'}),flush=True)
    finally:writer.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['smoke','pilot'],required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'manifest.yaml').exists():raise FileExistsError('Immutable run already started')
    config=json.loads(args.config.read_text());check_config(config)
    dirty=subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    if dirty:raise RuntimeError('Formal source must be clean')
    manifest=dict(phase=args.phase,status='running',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        private_repository='git@github.com:YUSIO/ByteTrack-MOT.git',branch='exp/193-trades-frozen-x',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_dirty=False,
        config=config,config_path=str(args.config),config_sha256=sha256(args.config),effective_argv=sys.argv,
        data_root=str(args.data_root),checkpoint=str(args.checkpoint),output=str(args.output),
        python_executable=sys.executable,python=sys.version,torch=str(torch.__version__),torchvision=str(torchvision.__version__),
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),platform=platform.platform(),
        gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),detector_seen_dev=True,test_access=False,parent_run=None,
        cuda_dcn_bitwise_reproducibility='not_guaranteed_atomic_backward',precision='fp32_no_tf32')
    write_json(args.output/'manifest.yaml',manifest)
    shutil.copyfile(args.config,args.output/'config.json')
    try:
        run(args,config,manifest);manifest['status']='completed'
    except BaseException as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        manifest['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(args.output/'manifest.yaml',manifest)


if __name__=='__main__':main()
