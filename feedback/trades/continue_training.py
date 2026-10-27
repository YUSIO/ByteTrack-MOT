"""Resume frozen TraDeS from epoch one through ten with immutable epoch evidence."""
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
from .model import FrozenTraDeS
from .data import TrainPairs,cva_loss
from .checkpoint import sha256,load_adapter,save_adapter
from .preflight import tensor_hashes
from .train import verify_inputs,check_gradients
from .diagnose import group,parameter_changes
from .validate import validate


def atomic_json(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def atomic_copy(source,target):
    temporary=target.with_name(target.name+'.tmp');shutil.copyfile(source,temporary);temporary.replace(target)


def validate_resume_config(config,parent):
    fixed=['batch_size','precision','seed','input_size','optimizer','lr','momentum','weight_decay',
           'cva_weight','gradient_clip','expected_fit_pairs','validation_modes','base_sha256',
           'history','dev_history','heat_shift','detector_seen_dev','inventories',
           'fit_sequences','dev_sequences','quarantine_sequences','gpu_uuid']
    for key in fixed:
        if config[key]!=parent[key]:raise ValueError(f'Resume changed fixed setting: {key}')
    if config['epochs']!=10 or config['start_epoch']!=2 or not config['resume'] or config['test_access']:
        raise ValueError('Only authorized epoch 2..10 continuation; no test')


def resume_order(seed,length,parent_order):
    generator=np.random.default_rng(seed)
    first=generator.permutation(length).tolist()
    if first!=[r['index'] for r in parent_order]:raise ValueError('Epoch1 order not reproducible')
    return generator


def restore_optimizer(optimizer,path,config):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['epoch']!=1 or state['seed']!=config['seed']:raise ValueError('Wrong optimizer epoch/seed')
    optimizer.load_state_dict(state['optimizer'])
    for group_config in optimizer.param_groups:
        for key in ['lr','momentum','weight_decay']:
            if group_config[key]!=config[key]:raise ValueError(f'Optimizer mismatch: {key}')
    torch.set_rng_state(state['torch_rng'])
    if torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state


def module_step_statistics(model,before):
    stats={}
    for name,p in model.adapter.named_parameters():
        row=stats.setdefault(group(name),dict(grad_sq=0.,update_sq=0.,parameter_sq=0.))
        if p.grad is not None:row['grad_sq']+=float(p.grad.detach().double().square().sum())
        row['update_sq']+=float((p.detach()-before[name]).double().square().sum())
        row['parameter_sq']+=float(p.detach().double().square().sum())
    return {name:{k.removesuffix('_sq')+'_l2':v**.5 for k,v in row.items()} for name,row in stats.items()}


def diagnostic_summary(path):
    rows=json.loads((path/'probes.json').read_text());result={}
    for part in ['fit','dev']:
        data=[r for r in rows if r['partition']==part];on=[r['conditions']['trained_eval_on'] for r in data]
        zero=[r['conditions']['trained_eval_support_zero'] for r in data]
        weights=[r['correspondence']['pairs'] for r in on]
        result[part]=dict(probes=len(data),pairs=sum(weights),
            off_detection_loss=float(np.mean([r['off_detection_loss'] for r in data])),
            on_detection_loss=float(np.mean([r['detection_loss'] for r in on])),
            cva_loss=float(np.mean([r['cva_loss'] for r in on])),
            joint_accuracy=float(np.average([r['correspondence']['joint_accuracy'] for r in on],weights=weights)),
            expected_epe_px=float(np.average([r['correspondence']['expected_epe_px'] for r in on],weights=weights)),
            foreground_relative_residual=float(np.mean([r['foreground']['relative_residual_rms'] for r in on])),
            support_zero_foreground_relative_change=float(np.mean([r['foreground']['relative_residual_change'] for r in zero])),
            support_zero_detection_loss=float(np.mean([r['detection_loss'] for r in zero])))
    return result


def checksum_tree(root):
    temporary=root/'outputs.sha256.tmp'
    with temporary.open('w') as stream:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.name not in ('outputs.sha256','outputs.sha256.tmp'):
                stream.write(sha256(p)+'  '+str(p.relative_to(root))+'\n')
    temporary.replace(root/'outputs.sha256')


def run(args,config,manifest):
    parent=json.loads((args.parent/'manifest.yaml').read_text())
    validate_resume_config(config,parent['config'])
    for name,expected in config['parent_files_sha256'].items():
        if sha256(args.parent/name)!=expected:raise ValueError(f'Parent changed: {name}')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=config['gpu_uuid'] or torch.cuda.device_count()!=1:
        raise RuntimeError('GPU0 UUID must be the only visible CUDA device')
    torch.set_num_threads(2);random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest['inputs']=verify_inputs(config,args.config,args.data_root,args.checkpoint,'pilot')
    atomic_json(args.output/'manifest.yaml',manifest)
    model=FrozenTraDeS(args.checkpoint,input_size=config['input_size']).cuda()
    initial={k:v.detach().cpu().clone() for k,v in model.adapter.state_dict().items()}
    metadata=load_adapter(model,args.parent/'last.pt',args.checkpoint)
    if metadata.get('epoch')!=1 or metadata.get('diagnostic_only'):raise ValueError('Require real epoch1 parent')
    frozen=tensor_hashes(model.detector)
    if frozen!=json.loads((args.parent/'detector_initial_sha256.json').read_text()):raise ValueError('Frozen detector differs from parent')
    optimizer=torch.optim.SGD(model.optimizer_parameters(),lr=config['lr'],momentum=config['momentum'],weight_decay=config['weight_decay'])
    restore_optimizer(optimizer,args.parent/'optimizer_last.pt',config)
    if {id(p) for g in optimizer.param_groups for p in g['params']}!={id(p) for p in model.adapter.parameters()}:raise RuntimeError('Optimizer ownership')
    samples=TrainPairs(args.data_root,config['fit_sequences'],config['input_size'])
    if len(samples)!=config['expected_fit_pairs']:raise ValueError('Dataset size changed')
    order_rng=resume_order(config['seed'],len(samples),json.loads((args.parent/'sample_order.json').read_text()))
    example=samples[0];x=example['image'][None].cuda();old=example['previous'][None].cuda();heat=example['heat'][None].cuda()
    model.eval()
    with torch.no_grad():baseline=model(x,enabled=False)['output'].clone()
    parent_metrics=json.loads((args.parent/'metrics.json').read_text())
    best_score=parent_metrics['validation']['on']['metrics']['AP50:95'];best_epoch=1
    history=[dict(epoch=1,source=str(args.parent),validation=parent_metrics['validation'],best_epoch=1)]
    atomic_copy(args.parent/'last.pt',args.output/'best.pt');atomic_copy(args.parent/'last.pt',args.output/'last.pt')
    atomic_json(args.output/'resume_evidence.json',dict(epoch1_loaded=True,optimizer_momentum_loaded=True,
        optimizer_state_entries=len(optimizer.state),epoch1_order_reconstructed=True,detector_frozen_tensors=len(frozen),
        adapter_metadata=metadata,parameter_names=[name for name,_ in model.adapter.named_parameters()]))
    writer=SummaryWriter(str(args.output/'tensorboard'));started=time.monotonic()
    try:
        for epoch in range(2,config['epochs']+1):
            out=args.output/'epochs'/f'epoch_{epoch:03d}';out.mkdir(parents=True,exist_ok=False)
            before_epoch={k:v.detach().cpu().clone() for k,v in model.adapter.state_dict().items()}
            order=order_rng.permutation(len(samples)).tolist()
            atomic_json(out/'sample_order.json',[dict(index=i,sequence=samples.index[i][0],frame=samples.index[i][1]) for i in order])
            epoch_started=time.monotonic();totals=dict(detection_loss=0.,cva_loss=0.,total_loss=0.);updates=skipped=0
            with (out/'training.csv').open('w',buffering=1) as log,(out/'module_steps.jsonl').open('w',buffering=1) as modules:
                fields=['epoch','step','global_step','sequence','frame','detection_loss','cva_loss','total_loss','grad_norm','pairs','collisions','skipped','elapsed_s']
                csvlog=csv.DictWriter(log,fieldnames=fields);csvlog.writeheader()
                for step,index in enumerate(order,1):
                    sample=samples[index];model.train();optimizer.zero_grad(set_to_none=True)
                    global_step=(epoch-1)*len(samples)+step
                    entry=dict(epoch=epoch,step=step,global_step=global_step,sequence=sample['sequence'],frame=sample['frame'],
                        pairs=sample['pair_stats']['used_pairs'],collisions=sample['pair_stats']['collision_excluded_pairs'],skipped=0)
                    measure=step<=2 or step%25==0 or step==len(order)
                    if sample['heat'].max()==0:
                        skipped+=1;entry.update(detection_loss=0.,cva_loss=0.,total_loss=0.,grad_norm=0.,skipped=1)
                    else:
                        result=model(sample['image'][None].cuda(),previous_images=sample['previous'][None].cuda(),
                            heat=sample['heat'][None].cuda(),targets=sample['targets'][None].cuda())
                        det=result['output']['total_loss'];aux=cva_loss(result['cva'],[sample['cva']]);loss=det+config['cva_weight']*aux
                        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss epoch {epoch} step {step}')
                        loss.backward();grad=check_gradients(model)
                        before={n:p.detach().clone() for n,p in model.adapter.named_parameters()} if measure else None
                        optimizer.step();updates+=1
                        entry.update(detection_loss=float(det.detach()),cva_loss=float(aux.detach()),total_loss=float(loss.detach()),grad_norm=grad)
                        if measure:
                            stats=module_step_statistics(model,before);modules.write(json.dumps(dict(epoch=epoch,step=step,global_step=global_step,groups=stats))+'\n')
                            for name,row in stats.items():
                                for key,v in row.items():writer.add_scalar('modules/'+name+'/'+key,v,global_step)
                        del result,det,aux,loss,before
                    for key in totals:totals[key]+=entry[key]
                    entry['elapsed_s']=time.monotonic()-epoch_started;csvlog.writerow(entry)
                    for key in ['detection_loss','cva_loss','total_loss','grad_norm']:writer.add_scalar('train/'+key,entry[key],global_step)
                    if measure:
                        progress=dict(phase='train',total_epochs=10,total_steps=len(order),**entry)
                        atomic_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True);writer.flush()
            if tensor_hashes(model.detector)!=frozen:raise RuntimeError('Frozen detector/BN changed')
            if any(not torch.isfinite(v).all() for v in model.adapter.state_dict().values()):
                raise FloatingPointError('Nonfinite adapter state at epoch boundary')
            model.eval();model.detector.head.decode_in_inference=True
            with torch.no_grad():
                off=model(x,enabled=False)['output'];empty=model(x,previous_images=old,heat=heat*0)['output']
                if not torch.equal(baseline,off) or not torch.equal(baseline,empty):raise RuntimeError('OFF/empty identity failed')
            save_adapter(model,out/'adapter.pt',args.checkpoint,dict(epoch=epoch,seed=config['seed'],phase='continuation',parent_run='run_002',diagnostic_only=False))
            torch.save(dict(optimizer=optimizer.state_dict(),epoch=epoch,seed=config['seed'],torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),numpy_order_rng=order_rng.bit_generator.state,
                parameter_names=[n for n,_ in model.adapter.named_parameters()]),out/'optimizer.pt')
            atomic_copy(out/'adapter.pt',args.output/'last.pt');atomic_copy(out/'optimizer.pt',args.output/'optimizer_last.pt')
            changes=dict(from_initial=parameter_changes(model,initial,epoch*len(samples),config['lr'],config['momentum'],config['weight_decay']),
                from_previous_epoch=parameter_changes(model,before_epoch,len(samples),config['lr'],config['momentum'],config['weight_decay']))
            atomic_json(out/'module_changes.json',changes)
            validation={}
            for mode in config['validation_modes']:
                atomic_json(args.output/'progress.json',dict(phase='validation',epoch=epoch,total_epochs=10,mode=mode))
                validation[mode]=validate(model,args.data_root,config['dev_sequences'],out/'validation'/mode,mode,'cuda:0')
                for key,v in validation[mode]['metrics'].items():writer.add_scalar('dev/'+mode+'/'+key,v,epoch)
            for name in config['dev_sequences']:
                if sha256(out/'validation/off'/name/'det.txt')!=sha256(args.parent/'validation/off'/name/'det.txt'):
                    raise RuntimeError('OFF cache differs from epoch1 baseline')
            atomic_json(args.output/'progress.json',dict(phase='diagnostic',epoch=epoch,total_epochs=10))
            command=[sys.executable,'-m','feedback.trades.diagnose','--data-root',str(args.data_root),'--checkpoint',str(args.checkpoint),
                '--adapter',str(out/'adapter.pt'),'--config',str(args.config),'--output',str(out/'diagnostic'),
                '--epoch',str(epoch),'--source-run',str(args.output.name)]
            atomic_json(out/'diagnostic_command.json',command)
            with (out/'diagnostic_stdout.log').open('w') as stdout,(out/'diagnostic_stderr.log').open('w') as stderr:
                subprocess.run(command,stdout=stdout,stderr=stderr,check=True)
            probe=diagnostic_summary(out/'diagnostic')
            for part,row in probe.items():
                for key,v in row.items():writer.add_scalar('probe/'+part+'/'+key,v,epoch)
            score=validation['on']['metrics']['AP50:95']
            if score>best_score:best_score=score;best_epoch=epoch;atomic_copy(out/'adapter.pt',args.output/'best.pt')
            model.detector.head.decode_in_inference=True
            if tensor_hashes(model.detector)!=frozen:raise RuntimeError('Detector changed during monitoring')
            summary=dict(epoch=epoch,updates=updates,skipped=skipped,mean_loss={k:v/len(samples) for k,v in totals.items()},
                training_seconds=entry['elapsed_s'],epoch_seconds=time.monotonic()-epoch_started,
                validation=validation,probe=probe,best_epoch=best_epoch,best_on_AP=best_score,
                detector_frozen_tensors=len(frozen),off_cache_matches_epoch1=True,off_empty_bitwise_identical=True,
                checkpoint_sha256=sha256(out/'adapter.pt'),optimizer_sha256=sha256(out/'optimizer.pt'))
            atomic_json(out/'metrics.json',summary);history.append(summary);atomic_json(args.output/'history.json',history)
            checksum_tree(out);writer.flush()
            print(json.dumps(dict(phase='epoch_complete',epoch=epoch,mean_loss=summary['mean_loss'],best_epoch=best_epoch,
                dev_on=validation['on']['metrics'],probe=probe)),flush=True)
        manifest.update(final_epoch=10,selected_epoch=best_epoch,selection='highest ON dev AP50:95 among epochs 1..10; earliest tie',elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/'metrics.json',dict(status='completed',final_epoch=10,selected_epoch=best_epoch,best_on_AP=best_score,
            epoch_history=history,performance_claim=False,test_access=False))
        atomic_json(args.output/'progress.json',dict(phase='completed',epoch=10,total_epochs=10,selected_epoch=best_epoch))
    finally:writer.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['config','data-root','checkpoint','parent','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'manifest.yaml').exists():raise FileExistsError('Immutable attempt already started')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Clean code required')
    config=json.loads(args.config.read_text());shutil.copyfile(args.config,args.output/'config.json')
    manifest=dict(phase='train_continuation',status='running',parent_run='run_002',start_epoch=2,total_epochs=10,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code_dirty=False,private_repository='git@github.com:YUSIO/ByteTrack-MOT.git',branch='exp/193-trades-frozen-x',
        config=config,config_sha256=sha256(args.config),effective_argv=sys.argv,working_directory=os.getcwd(),
        python_executable=sys.executable,python=sys.version,torch=str(torch.__version__),torchvision=str(torchvision.__version__),
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),platform=platform.platform(),
        gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),test_access=False,detector_seen_dev=True)
    atomic_json(args.output/'manifest.yaml',manifest)
    try:run(args,config,manifest);manifest['status']='completed'
    except BaseException as error:manifest.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        manifest['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();atomic_json(args.output/'manifest.yaml',manifest)


if __name__=='__main__':main()
