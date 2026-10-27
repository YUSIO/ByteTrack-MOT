"""Exp192 paired S1/T3 training, final-only checkpoints and natural calibration."""
from pathlib import Path
from collections import Counter
import argparse,csv,json,random,sys,time
import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score
SOURCE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(SOURCE))
from temporal_birth_common import Run,read,save,sha
from fctrack.temporal_birth import TemporalBirth,augment
from fctrack.verifier import load

def sampling_weights(rows):
    count=Counter((z['sequence'],z['frame'],z['owner']) for z in rows if int(z['label'])==1)
    w=np.array([1/count[z['sequence'],z['frame'],z['owner']] if int(z['label'])==1 else 1. for z in rows])
    for band in ['weak','strong']:
        for label in [0,1]:
            ids=np.array([z['band']==band and int(z['label'])==label for z in rows])
            if not ids.any():continue
            for seq in sorted({z['sequence'] for z in rows}):
                mask=ids&np.array([z['sequence']==seq for z in rows])
                if mask.any():w[mask]/=w[mask].sum()
            w[ids]/=w[ids].sum()
    return w/w.sum()

def measure(rows,p,threshold):
    y=np.array([int(z['label']) for z in rows]);valid=y>=0;y=y[valid];p=np.asarray(p)[valid];rr=[z for z,v in zip(rows,valid) if v]
    a=p>=threshold;good=a&(y==1);ref=np.array([int(z['strong_reference']) for z in rr])==1
    return dict(AP=float(average_precision_score(y,p)) if y.sum() else None,AUC=float(roc_auc_score(y,p)) if len(set(y))==2 else None,
        accepted=int(a.sum()),positive=int(good.sum()),negative=int((a&(y==0)).sum()),precision=float(good.sum()/a.sum()) if a.any() else 0.,
        positive_unique_GTframes=len({(z['sequence'],z['frame'],z['owner']) for z,v in zip(rr,good) if v}),positive_scenes=len({z['scene'] for z,v in zip(rr,good) if v}),
        strong_reference_count=int(ref.sum()),strong_reference_retention=float((a&ref).sum()/ref.sum()) if ref.any() else None)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--experiment',default='experiments/exp192_temporal_birth_gate');ap.add_argument('--config',default='configs/stage2.json');ap.add_argument('--arm',choices=['S1','T3'],required=True);ap.add_argument('--run',required=True);a=ap.parse_args()
    # A failed data gate must never create an optimizer or a training run.
    e=a.root/a.experiment;collection=read(e/a.config).get('collection_run','run_006')
    assert read(e/'results'/collection/'metrics.json')['G0_passed']
    assert torch.cuda.is_available(),'registered training requires idle CUDA GPU1; no CPU fallback'
    phase='temporal_birth_smoke' if read(e/a.config).get('smoke',False) else 'temporal_birth_training'
    with Run(a.root,a.experiment,a.config,a.run,phase) as run:
        plan=run.cfg['plan'];tc=plan['training'];seed=tc['first_seed'];device=torch.device('cuda:0')
        torch.set_num_threads(1);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
        run.manifest.update(arm=a.arm,seed=seed,device=str(device),physical_gpu_uuid=run.cfg['physical_gpu_uuid'],neural_weights_updated=True,calibration_read=True,checkpoint_selection='final_only',versions={'torch':torch.__version__,'numpy':np.__version__},gpu_name=torch.cuda.get_device_name(0))
        rows=list(csv.DictReader((run.e/'results'/collection/'samples.csv').open()));arrays=np.load(run.e/'results'/collection/'samples.npz')
        images=arrays['clips'];meta=arrays['metadata'];y=arrays['labels'];assert len(images)==len(rows)==len(y) and images.dtype==np.uint8
        train_ids=np.array([i for i,z in enumerate(rows) if z['group']=='fit' and int(z['label'])>=0]);train_rows=[rows[i] for i in train_ids]
        assert all(rows[i]['sequence'] in plan['data']['groups']['fit'] for i in train_ids)
        w=sampling_weights(train_rows);sampling_generator=torch.Generator().manual_seed(seed)
        count=tc['epochs']*tc['steps_per_epoch']*tc['batch_size'];sampled=torch.multinomial(torch.from_numpy(w),count,replacement=True,generator=sampling_generator).numpy();sampled=train_ids[sampled]
        if run.cfg.get('smoke'):
            positive=[i for i in train_ids if y[i]==1][:tc['batch_size']//2]
            negative=[i for i in train_ids if y[i]==0][:tc['batch_size']//2]
            assert len(positive)+len(negative)==tc['batch_size']
            sampled=np.tile(positive+negative,tc['epochs']*tc['steps_per_epoch'])
        aug_rng=torch.Generator().manual_seed(seed+1000)
        model=TemporalBirth();original=load('seed',run.root/plan['data']['seed_weights'],torch.device('cpu'));model.body.load_state_dict(original.body.state_dict());del original
        model.to(device);body=list(model.body.parameters());other=[p for n,p in model.named_parameters() if not n.startswith('body.')]
        opt=torch.optim.AdamW([{'params':body,'lr':tc['body_lr']},{'params':other,'lr':tc['fusion_head_lr']}],weight_decay=tc['weight_decay'])
        (run.r/'model').mkdir();np.save(run.r/'model/sampled_indices.npy',sampled)
        initial={n:p.detach().cpu().clone() for n,p in model.state_dict().items()};torch.save(initial,run.r/'model/initial_state.pt')
        history=[];start=time.perf_counter();step=0
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(str(run.r/'model/tensorboard'))
        for epoch in range(tc['epochs']):
            model.train();model.freeze_bn_statistics();frozen=epoch<tc['warmup_body_frozen_epochs']
            for p in body:p.requires_grad_(not frozen)
            losses=[]
            for k in range(tc['steps_per_epoch']):
                ids=sampled[step*tc['batch_size']:(step+1)*tc['batch_size']];step+=1
                x=torch.from_numpy(images[ids].astype(np.float32)/255);m=torch.from_numpy(meta[ids]);x,m=augment(x,m,aug_rng) if not run.cfg.get('smoke') else (x,m)
                labels=torch.from_numpy(y[ids].astype(np.float32)).to(device)
                opt.zero_grad(set_to_none=True);logits=model(x.to(device),m.to(device),a.arm)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels)
                assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),tc['gradient_clip']);assert torch.isfinite(norm)
                opt.step();losses.append(float(loss.item()))
            item=dict(epoch=epoch+1,loss=float(np.mean(losses)),body_frozen=frozen,steps=step,elapsed_seconds=time.perf_counter()-start)
            history.append(item);save(run.r/'history.json',history);writer.add_scalar('train/BCE',item['loss'],epoch+1);writer.add_scalar('train/body_frozen',int(frozen),epoch+1);writer.flush();print(json.dumps(item),flush=True)
        writer.close();assert step==tc['epochs']*tc['steps_per_epoch']
        torch.save(dict(state_dict=model.state_dict(),arm=a.arm,seed=seed,plan=plan,optimizer_steps=step),run.r/'model/final.pt')
        if run.cfg.get('smoke'):
            save(run.r/'metrics.json',dict(smoke=True,steps=step,loss=history[-1]['loss'],finite_loss_and_gradients=True,calibration_inference=False))
            return
        model.eval();pred=np.empty(len(rows),float)
        with torch.inference_mode():
            for start in range(0,len(rows),tc['batch_size']):
                end=min(start+tc['batch_size'],len(rows));xx=torch.from_numpy(images[start:end].astype(np.float32)/255).to(device);mm=torch.from_numpy(meta[start:end]).to(device)
                pred[start:end]=torch.sigmoid(model(xx,mm,a.arm)).cpu().numpy()
        with (run.r/'predictions.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=[*rows[0],'prediction']);writer.writeheader()
            for z,p in zip(rows,pred):writer.writerow(dict(z,prediction=float(p)))
        metrics={}
        for group in ['fit','calibration']:
            metrics[group]={}
            for threshold in plan['calibration']['threshold_grid']:
                per={}
                for band in ['all','weak','strong']:
                    ids=[i for i,z in enumerate(rows) if z['group']==group and (band=='all' or z['band']==band)]
                    per[band]=measure([rows[i] for i in ids],pred[ids],threshold)
                metrics[group][str(threshold)]=per
        feasible=[]
        for threshold in plan['calibration']['threshold_grid']:
            v=metrics['calibration'][str(threshold)];ok=True
            for band in ['all','weak']:
                r=v[band];ok &= r['precision']>=.8 and r['positive_unique_GTframes']>=10 and r['positive_scenes']>=3
            retain=v['all']['strong_reference_retention'];ok &= retain is not None and retain>=.95
            if ok:feasible.append(threshold)
        result=dict(arm=a.arm,seed=seed,metrics=metrics,selected_threshold=min(feasible) if feasible else None,feasible_thresholds=feasible,training_steps=step,model_parameters=sum(p.numel() for p in model.parameters()),checkpoint_sha256=sha(run.r/'model/final.pt'),initial_state_sha256=sha(run.r/'model/initial_state.pt'))
        save(run.r/'metrics.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
