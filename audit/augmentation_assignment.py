"""Read-only audit of the exact Exp141 transforms and native SimOTA.

No optimizer/backward. Uses native functions, observing locals without replacing
augmentation algebra. Unique IDs occupy the otherwise unused track-ID column;
images and the five training-label columns are checked against untouched code.
"""
import argparse
import contextlib
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import types
import numpy as np
import torch
import cv2
from yolox.exp import get_exp
from yolox.data.data_augment import TrainTransform, random_perspective
from yolox.data.datasets.mosaicdetection import MosaicDetection

MEAN=np.array([.485,.456,.406]);STD=np.array([.229,.224,.225])

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')

def state_hash(model):
    return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()}

def size_record(box,hw):
    x1,y1,x2,y2=map(float,box);h,w=hw
    bw,bh=x2-x1,y2-y1
    vw=max(0,min(x2,w)-max(0,x1));vh=max(0,min(y2,h)-max(0,y1))
    return dict(box=[x1,y1,x2,y2],width=bw,height=bh,short=min(bw,bh),sqrt_area=float(np.sqrt(max(0,bw*bh))),visible_width=vw,visible_height=vh,visible_short=min(vw,vh),visible_fraction=vw*vh/max(bw*bh,1e-30),center_inside=bool(0<=(x1+x2)/2<w and 0<=(y1+y2)/2<h))

class Observer:
    def __init__(self,wrapper,image_hashes):
        self.wrapper=wrapper;self.data=wrapper._dataset;self.hashes=image_hashes
        self.original_pull=self.data.pull_item
        self.data.pull_item=self.pull
        self.active=False
    def start(self,case,arm):
        self.case=case;self.arm=arm;self.rows={};self.next_uid=1;self.pulls=0;self.events={};self.calls={};self.active=True;self.pre_rng=None
    def pull(self,index):
        img,labels,info,id_=self.original_pull(index)
        if not self.active:return img,labels,info,id_
        rel='train/'+info[-1]
        if rel not in self.hashes:self.hashes[rel]=sha(Path(self.data.data_dir)/rel)
        role='single' if self.arm=='single' else ('mosaic' if self.pulls<4 else 'mixup')
        for j,row in enumerate(labels):
            uid=self.next_uid;self.next_uid+=1
            ratio=min(800/img.shape[0],1440/img.shape[1])
            self.rows[uid]=dict(case_id=self.case['case_id'],arm=self.arm,uid=uid,source_role=role,source_slot=self.pulls,source_index=int(index),source_image_id=int(id_[0]),source_file=info[-1],source_track_id=int(row[5]),source_row=j,stages={'raw':size_record(row[:4],img.shape[:2]),'base800':size_record(row[:4]*ratio,(800,1440))})
            labels[j,5]=uid
        self.pulls+=1
        return img,labels,info,id_
    def stage(self,name,labels,hw):
        for label in labels:
            self.rows[int(label[5])]['stages'][name]=size_record(label[:4],hw)
    def trace(self,frame,event,arg):
        code=frame.f_code
        if code not in (random_perspective.__code__,MosaicDetection.mixup.__code__,TrainTransform.__call__.__code__):return None
        v=frame.f_locals
        if event=='call':
            if code==random_perspective.__code__:
                self.calls[id(frame)]=v['targets'].copy();self.stage('mosaic_canvas',v['targets'],v['img'].shape[:2])
            elif code==TrainTransform.__call__.__code__:
                self.pre_rng=(random.getstate(),np.random.get_state());self.stage('pre_transform',v['targets'],v['image'].shape[:2])
            return self.trace
        if event=='return':
            if code==random_perspective.__code__:
                before=self.calls.pop(id(frame));self.events['affine_scale']=float(v['s']);self.events['rotation_degrees']=float(v['a'])
                if len(before):
                    full=before.copy();full[:,:4]=v['xy'];self.stage('affine_before_filter',full,(v['height'],v['width']))
                    kept=set(int(x) for x in arg[1][:,5])
                    for row,passed in zip(full,v['i']):
                        r=self.rows[int(row[5])]
                        if not passed:r['drop_stage']='affine_box_candidates'
                        elif int(row[5]) not in kept:r['drop_stage']='affine_outside_image'
                self.stage('affine_kept',arg[1],arg[0].shape[:2])
            elif code==MosaicDetection.mixup.__code__:
                self.events.update(mixup_jitter=float(v['jit_factor']),mixup_flip=bool(v['FLIP']),mixup_blended=bool(v['keep_list'].sum()>=1))
                ids=v['cp_labels'][:,5];labels=np.c_[v['cp_bboxes_transformed_np'],v['cp_labels'][:,4:6]]
                self.stage('mixup_before_filter',labels,(v['target_h'],v['target_w']))
                kept=set(int(x) for x in arg[1][:,5])
                for uid,passed in zip(ids,v['keep_list']):
                    r=self.rows[int(uid)]
                    if not passed:r['drop_stage']='mixup_box_candidates_wh5'
                    elif int(uid) not in kept:r['drop_stage']='mixup_outside_image'
            elif code==TrainTransform.__call__.__code__:
                self.events.update(transform_fallback=bool('mask_b' in v and not v['mask_b'].any()))
            return None
        return self.trace
    def generate(self,case,arm,rng=None,parity=False):
        self.start(case,arm)
        random.seed(case['augmentation_seed']);np.random.seed(case['augmentation_seed'])
        if rng is not None:random.setstate(rng[0]);np.random.set_state(rng[1])
        before=(random.getstate(),np.random.get_state())
        key=(tuple(case['input_hw']),case['dataset_index'],arm=='augmented')
        try:
            sys.settrace(self.trace)
            image,labels,_,_=self.wrapper[key]
        finally:sys.settrace(None);self.active=False
        after=(random.getstate(),np.random.get_state())
        if parity:
            random.setstate(before[0]);np.random.set_state(before[1])
            native,native_labels,_,_=self.wrapper[key]
            assert np.array_equal(image,native),'Instrumentation changed pixels'
            assert np.array_equal(labels[:,:5],native_labels[:,:5]),'Instrumentation changed labels'
            assert random.getstate()==after[0],'Instrumentation changed Python RNG'
            assert np.array_equal(np.random.get_state()[1],after[1][1]),'Instrumentation changed NumPy RNG'
        final=labels[labels[:,5]>0]
        for label in final:
            b=np.r_[label[1:3]-label[3:5]/2,label[1:3]+label[3:5]/2]
            r=self.rows[int(label[5])];r['stages']['final']=size_record(b,case['input_hw']);r['kept']=True
        for r in self.rows.values():
            if not r.get('kept'):
                r['kept']=False
                if 'drop_stage' not in r:r['drop_stage']='mosaic_canvas_outside' if 'pre_transform' not in r['stages'] else 'train_transform_filter_or_limit'
        return image,labels,final

class AssignmentObserver:
    def __init__(self,head):
        self.head=head;self.assign=head.get_assignments;self.inside=head.get_in_boxes_info;self.dynamic=head.dynamic_k_matching
        head.get_in_boxes_info=self.observe_inside;head.dynamic_k_matching=self.observe_dynamic;head.get_assignments=self.observe_assign
    def observe_inside(self,*args,**kwargs):
        result=self.inside(*args,**kwargs)
        self.candidate_mask=result[0].clone();self.valid=result[1].clone()
        return result
    def observe_dynamic(self,cost,ious,classes,num_gt,fg_mask):
        self.dynamic_k=ious.topk(min(10,ious.shape[1]),dim=1).values.sum(1).int().clamp_min(1)
        self.max_iou=ious.max(1).values
        return self.dynamic(cost,ious,classes,num_gt,fg_mask)
    def observe_assign(self,*args,**kwargs):
        values=inspect.signature(self.assign).bind(*args,**kwargs).arguments
        if values.get('mode','gpu')!='gpu':raise ValueError('CPU fallback forbidden; preserve failed attempt')
        result=self.assign(*args,**kwargs)
        _,fg,matched_iou,owner,npos=result
        stride=values['expanded_strides'][0]
        selected_stride=stride[fg]
        selected_columns=fg[self.candidate_mask].nonzero().flatten()
        selected_valid=self.valid[owner,selected_columns]
        # Convert small diagnostic vectors only. The original method owns matching.
        self.result=[]
        for j in range(values['num_gt']):
            mask=owner==j
            self.result.append(dict(positive_count=int(mask.sum()),dynamic_k=int(self.dynamic_k[j]),
                eligible_box_and_center=int(self.valid[j].sum()),max_candidate_iou=float(self.max_iou[j]),
                matched_iou_mean=float(matched_iou[mask].mean()) if mask.any() else None,
                matched_iou_max=float(matched_iou[mask].max()) if mask.any() else None,
                positives_outside_box_center=int((~selected_valid[mask]).sum()),
                positive_p3=int(((selected_stride==8)&mask).sum()),positive_p4=int(((selected_stride==16)&mask).sum()),positive_p5=int(((selected_stride==32)&mask).sum())))
        assert sum(r['positive_count'] for r in self.result)==npos
        return result

def make_dataset(root):
    os.environ['UAVSWARM_V1_ROOT']=str(root/'datasets/UAVSwarm-dataset-master')
    exp=get_exp('exps/example/uavswarm/yolox_x_uavswarmv1_full.py',None)
    return exp,exp.get_data_loader(batch_size=4,is_distributed=False).dataset

def save_preview(out,image,labels,case,events):
    rgb=np.ascontiguousarray(np.clip((image.transpose(1,2,0)*STD+MEAN)*255,0,255).astype(np.uint8))
    for row in labels[labels[:,5]>0]:
        c=row[1:3];wh=row[3:5];a=(c-wh/2).astype(int);b=(c+wh/2).astype(int)
        cv2.rectangle(rgb,tuple(a),tuple(b),(50,230,90),1)
    cv2.putText(rgb,f"case {case['case_id']}  affine scale={events.get('affine_scale',1):.3f}  H={case['input_hw'][0]}",(12,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,240,0),1)
    dest=out/'previews';dest.mkdir(exist_ok=True)
    cv2.imwrite(str(dest/f"case_{case['case_id']:04d}.jpg"),rgb[:,:,::-1],[cv2.IMWRITE_JPEG_QUALITY,92])

@torch.no_grad()
def execute(args,cfg,manifest):
    root=args.root;out=args.output
    for rel,digest in cfg['input_sha256'].items():assert sha(root/rel)==digest,rel
    cases=json.loads((root/cfg['cases']).read_text())['cases']
    if args.preflight:cases=[cases[i] for i in cfg['preflight_case_indices']]
    assert not cfg['test_access'] and cfg['optimizer_steps']==0
    cv2.setNumThreads(1);torch.set_num_threads(2)
    exp,aug=make_dataset(root);_,single=make_dataset(root)
    hashes={};augobs=Observer(aug,hashes);singleobs=Observer(single,hashes)
    runtime=dict(degrees=aug.degrees,translate=aug.translate,affine_scale=aug.scale,exp_mscale=exp.mscale,effective_mixup_scale=aug.mixup_scale,shear=aug.shear,perspective=aug.perspective,initial_inner_input=aug._dataset.input_dim,max_labels=aug.preproc.max_labels)
    dump(out/'effective_runtime.json',runtime)
    assert list(aug.mixup_scale)==[.5,1.5]
    manifest['effective_runtime']=runtime
    model=None
    if not args.preflight:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==cfg['gpu_uuid'] and torch.cuda.device_count()==1
        torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=exp.get_model().cuda().eval()
        ckpt=torch.load(root/cfg['checkpoint'],map_location='cpu',weights_only=False)
        model.load_state_dict(ckpt['model'],strict=True);del ckpt
        model.requires_grad_(False)
        # Only loss dispatch flags change; every BN remains eval.
        model.training=True;model.head.training=True;model.head.use_l1=False
        before=state_hash(model);assign=AssignmentObserver(model.head)
    all_events=[];all_original=[]
    for ann,info,file in aug._dataset.annotations:
        ratio=min(800/info[0],1440/info[1])
        for r in ann:
            all_original.append(dict(sequence=file.split('/')[0],raw=size_record(r[:4],info[:2]),base800=size_record(r[:4]*ratio,(800,1440))))
    with (out/'all_train_sizes.jsonl').open('w') as f:
        for r in all_original:f.write(json.dumps(r)+'\n')
    previews={k:0 for k in ('small','medium','large')}
    with (out/'objects.jsonl').open('w') as objects:
        for i,case in enumerate(cases):
            rng=None
            for arm,obs in [('augmented',augobs),('single',singleobs)]:
                image,labels,final=obs.generate(case,arm,rng=rng,parity=args.preflight or i<3)
                if arm=='augmented':rng=obs.pre_rng
                event=dict(**case,arm=arm,source_boxes=len(obs.rows),kept_boxes=len(final),**obs.events)
                if model is not None:
                    inputs=torch.from_numpy(image[None]).cuda().half()
                    targets=torch.from_numpy(labels[None,:,:5]).cuda().half()
                    assign.result=[]
                    with torch.autocast('cuda',dtype=torch.float16):loss=model(inputs,targets)
                    if not all(torch.isfinite(v).all() for v in loss.values() if torch.is_tensor(v)):raise ValueError('Nonfinite diagnostic loss')
                    if len(assign.result)!=len(final):raise ValueError('Assignment count differs from retained labels')
                    for r,assignment in zip(final,assign.result):obs.rows[int(r[5])]['assignment']=assignment
                    event['loss']={k:float(v) for k,v in loss.items()}
                    event['positive_count']=sum(a['positive_count'] for a in assign.result)
                for row in obs.rows.values():objects.write(json.dumps(row,allow_nan=False)+'\n')
                objects.flush();all_events.append(event)
                if arm=='augmented':
                    scale=obs.events['affine_scale'];group='small' if scale<.3 else ('large' if scale>1.5 else 'medium')
                    if previews[group]<2:
                        save_preview(out,image,labels,case,obs.events);previews[group]+=1
            if (i+1)%15==0 or args.preflight:print(json.dumps(dict(completed=i+1,total=len(cases),last_case=case['case_id'])),flush=True)
    dump(out/'samples.json',all_events);dump(out/'read_images.sha256.json',hashes)
    if model is not None:
        assert state_hash(model)==before,'Weights or BN changed'
        assert all(p.grad is None for p in model.parameters())
        manifest['all_weights_BN_unchanged']=True
    manifest.update(status='completed',cases=len(cases),arms=2,optimizer_steps=0,test_access=False,image_files_read=len(hashes),parity_passed=bool(args.preflight or len(cases)>=3))
    dump(out/'metrics.json',dict(cases=len(cases),forward_count=0 if args.preflight else 2*len(cases),read_images=len(hashes),all_train_boxes=len(all_original),training=False,test_access=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--preflight',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'manifest.yaml').exists():raise FileExistsError('Immutable execution already exists')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Clean source required')
    cfg=json.loads(args.config.read_text());manifest=dict(status='running',phase='augmentation_assignment_audit',preflight=args.preflight,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_dirty=False,private_repository='git@github.com:YUSIO/ByteTrack-MOT.git',branch='exp/141-augmentation-audit',base_commit='ee7b677cc25c8070f45ce89bf63b532089dc7655',checkpoint=cfg['checkpoint'],checkpoint_sha256=cfg['input_sha256'][cfg['checkpoint']],config_sha256=sha(args.config),python=sys.version,python_executable=sys.executable,torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__,gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),argv=sys.argv,training=False,test_access=False,optimizer_steps=0,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    dump(args.output/'manifest.yaml',manifest);dump(args.output/'config.json',cfg)
    try:execute(args,cfg,manifest)
    except BaseException as e:manifest.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:manifest['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();dump(args.output/'manifest.yaml',manifest)

if __name__=='__main__':main()
