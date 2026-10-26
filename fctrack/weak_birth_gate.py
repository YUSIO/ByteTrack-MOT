"""GT-free, fixed-parameter acceptance score for supplemental weak births."""
import numpy as np

FEATURE_NAMES=('det_logit','seed_logit','agreement','center_shift','log_area_ratio',
               'log_raw_area','proposal_cover','regressed_cover')

def features(score,p,raw,reg,proposal_cover,regressed_cover):
    raw=np.asarray(raw,float).reshape(-1,4);reg=np.asarray(reg,float).reshape(-1,4)
    assert raw.shape==reg.shape
    area=np.maximum(raw[:,2:].prod(1),1e-6);ra=np.maximum(reg[:,2:].prod(1),1e-6)
    lo=np.maximum(raw[:,:2],reg[:,:2]);hi=np.minimum(raw[:,:2]+raw[:,2:],reg[:,:2]+reg[:,2:])
    inter=np.maximum(hi-lo,0).prod(1);agreement=inter/np.maximum(area+ra-inter,1e-6)
    shift=np.linalg.norm((raw[:,:2]+raw[:,2:]/2)-(reg[:,:2]+reg[:,2:]/2),axis=1)/np.sqrt(area)
    def logit(x):
        x=np.clip(np.asarray(x,float),1e-6,1-1e-6);return np.log(x/(1-x))
    x=np.column_stack((logit(score),logit(p),agreement,shift,np.log(ra/area),np.log(area),proposal_cover,regressed_cover))
    if not np.isfinite(x).all():raise ValueError('non-finite weak gate features')
    return x

def predict(model,x):
    indices=[FEATURE_NAMES.index(n) for n in model['feature_names']]
    x=np.asarray(x,float)[:,indices]
    mean=np.asarray(model['mean'],float);std=np.asarray(model['std'],float);coef=np.asarray(model['coef'],float)
    if not (mean.shape==std.shape==coef.shape==(len(indices),)) or np.any(std<=0):
        raise ValueError('invalid frozen weak gate')
    z=((x-mean)/std)@coef+float(model['intercept'])
    return 1/(1+np.exp(-np.clip(z,-60,60)))
