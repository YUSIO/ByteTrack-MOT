"""Read-only interventions after a fixed-budget real-loss replay; no optimizer steps."""
import torch
import torch.nn.functional as F
from .learning_gate import batch


@torch.no_grad()
def audit_history(model, samples, device):
    rows=[]
    m=model.feedback
    for s in samples:
        x,y,c=batch([s],device);_,_,wrong=batch([s],device,'shifted')
        h,w=x.shape[-2:];boxes=c[0]['boxes'];ids=c[0]['ids']
        q=m.encode_query(boxes,ids,h,w);patch=m.patch_embed(x).flatten(2).transpose(1,2)[0]
        a=m.attention(q,patch,h,w,boxes);p=a@m.value(patch);g=m.backproject(a,p)
        delta=m(x,boxes,ids)-x;shift=m(x,wrong[0]['boxes'],ids)-x
        match=shift*(delta.square().mean().sqrt()/shift.square().mean().sqrt().clamp_min(1e-30))
        interventions={'off':torch.zeros_like(delta),'correct':delta,'shifted':shift,
                       'correct_clip_0.5':delta.clamp(-.5,.5),'correct_half':delta*.5,
                       'shifted_rms_matched':match}
        losses={}
        for name,d in interventions.items():
            out=model(x+d,y,[None]);losses[name]={k:float(out[k]) for k in ['total_loss','iou_loss','conf_loss','cls_loss']}
        centers=boxes[:,:2]+boxes[:,2:]/2
        rr,cc=torch.meshgrid(torch.arange(h//m.patch,device=device),torch.arange(w//m.patch,device=device),indexing='ij')
        xy=torch.stack((cc+.5,rr+.5),-1).reshape(-1,2)*m.patch
        dist=(xy[None]-centers[:,None]).norm(dim=-1)
        peakdist=dist[torch.arange(len(boxes),device=device),a.argmax(-1)]
        square=((xy[None]-centers[:,None]).abs()<=32).all(-1)
        perturb=boxes+boxes.new_tensor([1.,0.,0.,0.]);qj=m.encode_query(perturb,ids,h,w)
        aj=m.attention(qj,patch,h,w,perturb)
        effective=a.shape[1]*a.sum(0)/len(boxes)
        rows.append({'sequence':s['sequence'],'frame':s['frame'],'N':len(boxes),
                     'losses':losses,'delta_rms':float(delta.square().mean().sqrt()),'delta_max':float(delta.abs().max()),
                     'shifted_delta_rms':float(shift.square().mean().sqrt()),'clip_fraction':float((delta.abs()>.5).float().mean()),
                     'attention_peak_distance_mean_px':float(peakdist.mean()),'attention_peak_distance_max_px':float(peakdist.max()),
                     'attention_mass_in_history_32px_square':float((a*square).sum(-1).mean()),
                     'attention_max':float(a.max()),'old_M_mean_coverage_max':float(effective.max()),
                     'query_1px_cosine_mean':float(F.cosine_similarity(q,qj).mean()),
                     'attention_1px_L1_mean':float((a-aj).abs().sum(-1).mean()),
                     'prompt_rms':float(p.square().mean().sqrt()),'grid_rms':float(g.square().mean().sqrt())})
    return rows
