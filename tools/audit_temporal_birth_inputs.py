"""Post-hoc saved T3 input-dependence probe. No optimization or model selection."""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np,torch
SOURCE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(SOURCE))
from temporal_birth_common import Run,save
from fctrack.temporal_birth import TemporalBirth

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',default='configs/failure_probe.json');p.add_argument('--run',default='run_011');a=p.parse_args()
 with Run(a.root,'experiments/exp192_temporal_birth_gate',a.config,a.run,'posthoc_saved_model_input_dependence') as run:
  torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
  device=torch.device(run.cfg.get('device','cuda:0'))
  if device.type=='cuda':assert torch.cuda.is_available()
  data=np.load(run.e/'results/run_007/samples.npz');rows=list(csv.DictReader((run.e/'results/run_007/samples.csv').open()))
  reference=np.array([float(z['prediction']) for z in csv.DictReader((run.e/'results/run_010/predictions.csv').open())])
  checkpoint=torch.load(run.e/'results/run_010/model/final.pt',map_location='cpu',weights_only=False)
  model=TemporalBirth();model.load_state_dict(checkpoint['state_dict']);model.to(device).eval();pred={name:[] for name in ['real_history','repeat_current']}
  with torch.inference_mode():
   for start in range(0,len(rows),64):
    clips=torch.from_numpy(data['clips'][start:start+64].astype(np.float32)/255).to(device);metadata=torch.from_numpy(data['metadata'][start:start+64]).to(device)
    for name,arm in [('real_history','T3'),('repeat_current','S1')]:pred[name].extend(torch.sigmoid(model(clips,metadata,arm)).cpu().numpy().tolist())
  pred={k:np.asarray(v,float) for k,v in pred.items()};error=float(np.abs(pred['real_history']-reference).max())
  weak=np.array([z['band']=='weak' for z in rows]);changed_weak=int(np.sum((pred['real_history'][weak]>=.95)!=(reference[weak]>=.95)))
  if device.type=='cuda':assert error<=1e-6,error
  else:assert changed_weak==0,changed_weak
  cohorts={}
  S1=np.array([float(z['prediction']) for z in csv.DictReader((run.e/'results/run_009/predictions.csv').open())])
  for name,group,label in [('cal_weak_positive','calibration',1),('cal_weak_negative','calibration',0),('fit_weak_positive','fit',1),('fit_weak_negative','fit',0)]:
   ids=[i for i,z in enumerate(rows) if z['group']==group and z['band']=='weak' and int(z['label'])==label]
   cohorts[name]=dict(rows=len(ids),real_above_095=int((pred['real_history'][ids]>=.95).sum()),repeat_above_095=int((pred['repeat_current'][ids]>=.95).sum()))
  ids=[i for i,z in enumerate(rows) if z['group']=='calibration' and z['band']=='weak' and int(z['label'])==0 and reference[i]>=.95 and S1[i]<.95]
  cohorts['T3_only_wrong_accepts']=dict(rows=len(ids),repeat_still_above_095=int((pred['repeat_current'][ids]>=.95).sum()),repeat_below_095=int((pred['repeat_current'][ids]<.95).sum()))
  with (run.r/'predictions.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=['sample_index','group','sequence','frame','index','label','band','original_S1','real_history','repeat_current']);w.writeheader()
   for i,z in enumerate(rows):w.writerow(dict(sample_index=i,**{k:z[k] for k in ['group','sequence','frame','index','label','band']},original_S1=float(S1[i]),**{k:float(v[i]) for k,v in pred.items()}))
  result=dict(device=str(device),real_history_vs_original_GPU_max_error=error,weak_threshold_decision_changes=changed_weak,cohorts=cohorts,interpretation='conditional_input_dependence_only_repeat_current_is_not_a_trained_or_selected_deployment_model',threshold_used_only_for_posthoc_description=.95)
  run.manifest.update(device=str(device),physical_gpu_uuid=run.cfg['physical_gpu_uuid'] if device.type=='cuda' else None,optimizer_executed=False,training=False,calibration_read=True,original_training_commit=run.cfg['training_commit'])
  save(run.r/'metrics.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
