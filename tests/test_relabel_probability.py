import unittest
import numpy as np
from fctrack.relabel_probability import evidence,calibrate

class ProbabilityTests(unittest.TestCase):
    def model(self):
        return dict(beta0=1.,beta1=2.,sigma2=.5,tau=6.,prior=.4,rho=.5)
    def test_uninformative_evidence_is_neutral(self):
        m=self.model()
        for d in [0.,3.,6.]:
            e=evidence(d,None,m)
            self.assertEqual(e['bayes_factor'],1.)
            self.assertAlmostEqual(e['posterior'],m['prior'])
            self.assertFalse(e['rejected'])
        self.assertAlmostEqual(evidence(6.,1e8,m)['bayes_factor'],1.,places=12)
    def test_risk_rule_and_distance_order(self):
        m=self.model();cf=(1-m['prior'])/(m['prior']*m['rho'])
        values=[]
        for d in np.linspace(0,6,101):
            e=evidence(float(d),.1,m);p=e['posterior']
            self.assertTrue(0<=p<=1)
            self.assertEqual(e['rejected'],cf*p<1-p)
            values.append(p)
        self.assertTrue(np.all(np.diff(values)<=0))
    def test_calibration_excludes_unknown_and_bounds_harm(self):
        rows=[dict(label='correct',distance=.2,spread=.1),dict(label='correct',distance=.8,spread=.4),dict(label='wrong',distance=5.,spread=2.),dict(label='wrong',distance=4.,spread=1.)]
        m=calibrate(rows)
        self.assertEqual(m,calibrate(rows+[dict(label='unknown',distance=100.,spread=200.)]))
        self.assertTrue(m['beta1']>=0 and m['rho']<=.5)
        self.assertFalse(any(evidence(r['distance'],r['spread'],m)['rejected'] for r in rows if r['label']=='correct'))

if __name__=='__main__':unittest.main()
