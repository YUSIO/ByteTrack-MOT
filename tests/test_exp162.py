import copy
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
import torch
from feedback.weak import (CausalState, DetectionTracker, WeakAdapter, WeakDetector,
                           heat_from_record, render, select_rows)
from feedback.weak_history import train_batch, SnapshotPairs, sha256


def det(x, s, y=0.):
    return [x,y,x+10,y+10,s]


def rows(values):
    return np.asarray(values,np.float64).reshape(-1,5)


class Eligibility(unittest.TestCase):
    def test_strict_band_unmatched(self):
        d=rows([det(i*30,s) for i,s in enumerate([.1,.100001,.599999,.6,.600001,.699999,.7])])
        tr=DetectionTracker(); out,a=tr.update(d,(800,1440)); t,u=select_rows(d,a)
        self.assertEqual(u,[1,2]); self.assertEqual(t,[6])
        self.assertEqual(a[6]['match_stage'],'birth')

    def test_matched_low_not_U_and_unconfirmed_high_not_T(self):
        tr=DetectionTracker();tr.update(rows([det(0,.9)]),(800,1440))
        d=rows([det(0,.3),det(100,.8),det(200,.3),det(300,.65)])
        _,a=tr.update(d,(800,1440));t,u=select_rows(d,a)
        self.assertEqual(t,[0]);self.assertEqual(u,[2])
        self.assertEqual(a[0]['match_stage'],'low')
        self.assertEqual(a[1]['match_stage'],'birth');self.assertFalse(a[1]['output_confirmed'])
        self.assertIsNone(a[3]['consumed_by'])

    def test_lost_low_not_consumed(self):
        tr=DetectionTracker();tr.update(rows([det(0,.9)]),(800,1440))
        tr.update(rows([]),(800,1440))
        d=rows([det(0,.3)]);_,a=tr.update(d,(800,1440))
        self.assertEqual(select_rows(d,a),([],[0]))

    def test_high_reactivation(self):
        tr=DetectionTracker();out,_=tr.update(rows([det(0,.9)]),(800,1440))
        identity=out[0][0];tr.update(rows([]),(800,1440))
        out,a=tr.update(rows([det(0,.9)]),(800,1440))
        self.assertEqual(out[0][0],identity);self.assertTrue(a[0]['output_confirmed'])

    def test_unconfirmed_match(self):
        tr=DetectionTracker();tr.update(rows([]),(800,1440))
        tr.update(rows([det(100,.9)]),(800,1440))
        _,a=tr.update(rows([det(100,.9)]),(800,1440))
        self.assertEqual(a[0]['match_stage'],'unconfirmed');self.assertTrue(a[0]['output_confirmed'])

    def test_duplicate_input_has_exact_row_owner(self):
        tr=DetectionTracker();tr.update(rows([det(0,.9)]),(800,1440))
        d=rows([det(0,.4),det(0,.4)]);_,a=tr.update(d,(800,1440))
        t,u=select_rows(d,a)
        self.assertEqual(len(t),1);self.assertEqual(len(u),1);self.assertEqual(set(t+u),{0,1})

    def test_hook_does_not_change_tracking(self):
        on,off=DetectionTracker(True),DetectionTracker(False)
        rng=np.random.default_rng(162)
        for frame in range(50):
            d=rows([det(i*30+frame*.3,float(rng.choice([.05,.1,.3,.6,.65,.8,.95]))) for i in range(8)])
            a,_=on.update(d,(800,1440));b,_=off.update(d,(800,1440))
            self.assertEqual(a,b)
            for field in ['tracked_stracks','lost_stracks','removed_stracks']:
                left,right=getattr(on.tracker,field),getattr(off.tracker,field)
                self.assertEqual([x.track_id for x in left],[x.track_id for x in right])
                for x,y in zip(left,right):
                    np.testing.assert_array_equal(x.mean,y.mean)
                    np.testing.assert_array_equal(x.covariance,y.covariance)

    def test_canonical_Exp141_tracker_equivalence(self):
        from feedback.weak_check import reference_tracker_class
        tr, ref = DetectionTracker(), DetectionTracker(False,reference_tracker_class())
        rng=np.random.default_rng(163)
        for frame in range(30):
            d=rows([det(i*30+frame*.5,float(rng.choice([.05,.3,.65,.9]))) for i in range(5)])
            left,_=tr.update(d,(800,1440));right,_=ref.update(d,(800,1440))
            self.assertEqual(left,right)


class Causality(unittest.TestCase):
    def test_future_and_cross_sequence_rejected(self):
        s=CausalState('TU');s.reset('a')
        for name,frame in [('a',0),('a',2),('b',1)]:
            with self.assertRaises(ValueError):s.snapshot(name,frame,'x')

    def test_one_frame_empty_overwrite_and_reset(self):
        s=CausalState('TU');s.reset('a');d=rows([det(32,.3)])
        _,a=DetectionTracker().update(d,(800,1440));s.advance('a',1,d,a)
        self.assertGreater(s.heat('a',2,1.,(100,180)).sum(),0)
        s.advance('a',2,rows([]),[]);self.assertEqual(s.heat('a',3,1.,(100,180)).sum(),0)
        s.reset('b');self.assertEqual(s.snapshot('b',1,'x')['source_frame'],0)

    def test_snapshot_copy_no_future_mutation(self):
        s=CausalState('TU');s.reset('a');h=s.snapshot('a',1,'x')
        d=rows([det(32,.3)]);_,a=DetectionTracker().update(d,(800,1440));s.advance('a',1,d,a)
        d[0]=0;self.assertEqual(h['rows'],[]);self.assertEqual(s.rows[0,4],.3)

    def test_snapshot_epoch_validation(self):
        s=CausalState('TU');s.reset('a');h=s.snapshot('a',1,'sha')
        for kw in [{'expected_snapshot':'bad'},{'expected_sequence':'b'},{'expected_frame':2}]:
            with self.assertRaises(ValueError):heat_from_record(h,1,(100,180),**kw)
        h['source_frame']=1
        with self.assertRaises(ValueError):heat_from_record(h,1,(100,180))

    def test_first_frame_gt_like_prior_rejected(self):
        h=CausalState('TU').snapshot('unset',1,'x');h['rows']=[det(32,.9)]
        with self.assertRaises(ValueError):heat_from_record(h,1,(100,180))


class Rendering(unittest.TestCase):
    def test_score_amplitude_max_and_coordinates(self):
        d=rows([[28,28,36,36,.4],[28,28,36,36,.2]])
        h=render(d,1.)
        self.assertAlmostEqual(float(h[0,4,4]),.4,places=6)
        self.assertEqual(float(h[0,4,8]),0.)
        self.assertAlmostEqual(float(render(d,2.)[0,8,8]),.4,places=6)

    def test_fractional_centres_and_padding(self):
        d=rows([[29,28,37,36,.4]])
        h=render(d,1.)
        self.assertLess(float(h.max()),.4)
        self.assertTrue(torch.equal(render(d,1.,padding=(8,0))[:,:,1:],h[:,:,:-1]))

    def test_empty_offscreen_and_clipped(self):
        self.assertEqual(render(rows([]),1.).sum(),0)
        self.assertEqual(render(rows([det(-1000,.3)]),1.).sum(),0)
        self.assertGreater(render(rows([[-4,-4,4,4,.3]]),1.).sum(),0)

    def test_adapter_zero_input_and_zero_init(self):
        a=WeakAdapter();h=torch.rand(1,1,8,8,requires_grad=True)
        self.assertEqual(sum(p.numel() for p in a.parameters()),5264)
        self.assertEqual(a(h).abs().max(),0)
        with torch.no_grad():a.output.weight.normal_()
        self.assertEqual(a(torch.zeros_like(h)).abs().max(),0)
        a(h).sum().backward();self.assertIsNone(h.grad)

    def test_single_candidate_nms(self):
        from yolox.utils import postprocess
        x=torch.tensor([[[10.,10.,4.,4.,.9,.9]]])
        self.assertEqual(postprocess(x,1,.001,.7)[0].shape,(1,7))

    def test_T_TU_only_source_changes(self):
        s=CausalState('TU');s.reset('a');d=rows([det(32,.9),det(100,.3)])
        _,a=DetectionTracker().update(d,(800,1440));s.advance('a',1,d,a)
        h=s.snapshot('a',2,'x');tu=heat_from_record(h,1.,(100,180));h['arm']='T'
        t=heat_from_record(h,1.,(100,180))
        self.assertTrue(torch.all(tu>=t));self.assertGreater((tu-t).sum(),0)


class TrainingHistory(unittest.TestCase):
    def fixture(self, root):
        path=root/'train/UAVSwarm-03';(path/'img1').mkdir(parents=True);(path/'gt').mkdir()
        (path/'seqinfo.ini').write_text('[Sequence]\nname=UAVSwarm-03\nimDir=img1\nframeRate=30\nseqLength=2\nimWidth=32\nimHeight=32\nimExt=.jpg\n')
        (path/'gt/gt.txt').write_text('1,1,1,1,5,5,1,1,1\n2,1,2,2,5,5,1,1,1\n')
        for frame in (1,2):cv2.imwrite(str(path/'img1'/f'{frame:06d}.jpg'),np.zeros((32,32,3),np.uint8))
        cache=root/'cache';cache.mkdir();state=CausalState('TU');state.reset('UAVSwarm-03')
        first=state.snapshot('UAVSwarm-03',1,'snapshot')
        d=rows([det(10,.3)]);_,audit=DetectionTracker().update(d,(32,32))
        state.advance('UAVSwarm-03',1,d,audit);second=state.snapshot('UAVSwarm-03',2,'snapshot')
        file=cache/'UAVSwarm-03.jsonl';file.write_text(json.dumps(first)+'\n'+json.dumps(second)+'\n')
        manifest={'snapshot_sha256':'snapshot','arm':'TU','sequences':['UAVSwarm-03'],
                  'gt_access':False,'files':{'UAVSwarm-03':sha256(file)}}
        (cache/'manifest.json').write_text(json.dumps(manifest))
        return path,cache

    def test_GT_changes_targets_not_feedback_and_shuffle_keeps_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,cache=self.fixture(root)
            pairs=SnapshotPairs(root,['UAVSwarm-03'],cache,'snapshot','TU')
            second=pairs[1];first=pairs[0]
            self.assertEqual(first[1].sum(),0);self.assertGreater(second[1].sum(),0)
            self.assertFalse(second[1].requires_grad)
            (path/'gt/gt.txt').write_text('1,1,20,20,5,5,1,1,1\n2,1,22,22,5,5,1,1,1\n')
            changed=SnapshotPairs(root,['UAVSwarm-03'],cache,'snapshot','TU')[1]
            self.assertTrue(torch.equal(second[1],changed[1]))
            self.assertFalse(torch.equal(second[2],changed[2]))
            self.assertTrue(torch.equal(pairs[1][1],second[1]))

    def test_wrong_epoch_arm_and_tampered_cache_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,cache=self.fixture(root)
            for snapshot,arm in [('stale','TU'),('snapshot','T')]:
                with self.assertRaises(ValueError):SnapshotPairs(root,['UAVSwarm-03'],cache,snapshot,arm)
            with (cache/'UAVSwarm-03.jsonl').open('a') as f:f.write('\n')
            with self.assertRaises(ValueError):SnapshotPairs(root,['UAVSwarm-03'],cache,'snapshot','TU')


if __name__=='__main__':unittest.main()
