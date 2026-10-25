"""Fixed Exp172 binary post-selection model; no identity reassignment."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

def evidence(distance, spread, model):
    a = 0. if spread is None else float(expit(model['beta0']-model['beta1']*np.log1p(spread)))
    variance = model['sigma2']
    z = -np.expm1(-model['tau']**2/(2*variance))
    g = float(model['tau']**2/(2*variance*z)*np.exp(-distance**2/(2*variance)))
    bf = 1. + a*(g-1.)
    prior = model['prior']
    posterior = prior*bf/(1.-prior+prior*bf)
    return dict(reliability=a, geometric_bayes_factor=g, bayes_factor=bf,
                posterior=float(posterior), rejected=bool(bf < model['rho']))

def calibrate(rows):
    known = [r for r in rows if r['label'] in ('correct','wrong')]
    available = [r for r in known if r['spread'] is not None]
    x = np.log1p([r['spread'] for r in available])
    y = np.array([r['label']=='correct' for r in available],float)
    assert len(y) and 0 < y.sum() < len(y)
    def objective(beta):
        b0,b1 = beta
        logits = b0-b1*x
        diff = expit(logits)-y
        loss = np.logaddexp(0,logits).sum()-(y*logits).sum()+.5*b1*b1
        gradient = np.array([diff.sum(),-(diff*x).sum()+b1])
        return float(loss),gradient
    fit = minimize(objective,[0.,0.],jac=True,bounds=[(None,None),(0,None)],method='L-BFGS-B')
    assert fit.success,fit.message
    correct = [r for r in known if r['label']=='correct']
    n = len(correct)
    model = dict(beta0=float(fit.x[0]),beta1=float(fit.x[1]),
                 sigma2=max(float(np.mean([r['distance']**2 for r in correct]))/2,1e-6),
                 tau=6.,prior=(n+1)/(len(known)+2),rho=0.,fit_correct=n,
                 fit_wrong=len(known)-n,fit_available=len(available),fit_loss=float(fit.fun))
    factors = sorted(evidence(r['distance'],r['spread'],model)['bayes_factor'] for r in correct)
    model['rho_fit'] = factors[int(np.floor(.05*n))]
    model['rho'] = min(model['rho_fit'],.5)
    model['false_merge_cost'] = 1.
    model['false_split_cost'] = (1-model['prior'])/(model['prior']*model['rho'])
    return model
