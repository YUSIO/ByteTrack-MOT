import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fctrack.output_verification import _parse_row
from fctrack.peer_relation import (ARMS, SCORE_COLUMNS, QueryHead, RelationHead, apply,
    bank_from_queries, crop_contains_query, detector_coordinates, extract, frame_support,
    load_extraction, read_queries, read_rows, sample_indices, select_cross, sha256,
    write_queries, Evidence, load_encoder)


CONFIG = Path(__file__).resolve().parents[3] / "experiments/exp154_peer_rgb_relation_v1/configs/dev.json"


def row(frame, identity, box, index=0):
    raw = f"{frame},{identity},{','.join(str(x) for x in box)},.9,-1,-1,-1\n".encode()
    return _parse_row(raw, index, "synthetic")


class PeerRelationTests(unittest.TestCase):
    def test_maturity_previous_five_frames_and_prefix_causality(self):
        history, records = {}, []
        det = np.array([[0, 0, 10, 10, .9], [100, 0, 10, 10, .9]])
        for frame in range(1, 7):
            records.append(frame_support([row(frame, 1, det[0, :4], 2*(frame-1)),
                                           row(frame, 2, det[1, :4], 2*(frame-1)+1)], det, history))
        self.assertTrue(all(not q["eligible"] for values in records[:5] for q in values))
        self.assertEqual(records[5][0]["prior_streak"], 5)
        self.assertEqual(records[5][0]["support_indices"], [11])
        prefix = json.dumps(records)
        frame_support([row(8, 1, det[0, :4], 12)], det, history)
        self.assertEqual(history[1], (8, 1))
        self.assertEqual(json.dumps(records), prefix)

    def test_score_filter_before_argmax_iou_and_original_index_dedup(self):
        rows = [row(6, 1, [0, 0, 10, 10]), row(6, 2, [.1, 0, 10, 10], 1),
                row(6, 3, [100, 0, 10, 10], 2)]
        det = np.array([[0, 0, 10, 10, .1], [.2, 0, 10, 10, .9], [100, 0, 10, 10, .9]])
        info = frame_support(rows, det, {i: (5, 5) for i in (1, 2, 3)})
        self.assertEqual(info[0]["det_index"], 1)
        self.assertEqual(info[1]["det_index"], 1)
        self.assertFalse(info[0]["anchor_eligible"])
        self.assertTrue(info[1]["anchor_eligible"])
        self.assertEqual(info[2]["support_indices"], [1])

    def test_crop_exclusion_touch_and_detector_quantization(self):
        self.assertTrue(crop_contains_query([0, 0, 10, 10], [19.99, 0, 10, 10]))
        self.assertFalse(crop_contains_query([0, 0, 10, 10], [20, 0, 10, 10]))
        det = np.array([[0, 0, 10, 10, .9], [15, 0, 10, 10, .9]])
        info = frame_support([row(6, 1, det[0,:4]), row(6, 2, det[1,:4], 1)], det, {1:(5,5),2:(5,5)})
        self.assertFalse(any(q["eligible"] for q in info))
        cache = np.array([[1,-1,123.123456,56.234567,7.345678,8.456789,.81234567,-1,-1,-1]])
        actual = detector_coordinates(cache)
        xyxy = np.r_[cache[0,2:4],cache[0,2:4]+cache[0,4:6]].astype(np.float32)
        np.testing.assert_array_equal(actual[0,:2], xyxy[:2])
        np.testing.assert_array_equal(actual[0,2:4]+actual[0,:2], xyxy[2:])
        self.assertEqual(actual[0,4], float(np.float32(cache[0,6])))

    def test_cross_bank_train_only_selection_and_count(self):
        def anchor(index, frame, identity, score):
            return dict(row_index=index, frame=frame,id=identity,det_score=score,anchor_eligible=True)
        queries={"train1":[anchor(0,6,1,.9),anchor(1,7,1,.95)],
                 "train2":[anchor(0,6,1,.8),anchor(1,6,2,.9)],
                 "dev":[anchor(0,6,99,1.)]}
        bank=bank_from_queries(queries,["train1","train2"])
        self.assertEqual(len(bank),3)
        self.assertEqual(bank[0]["row_index"],1)
        q=dict(frame=6,id=1,support_indices=[0,1])
        refs=select_cross(bank,"train1",q)
        self.assertEqual({r["sequence"] for r in refs},{"train2"})
        self.assertEqual(refs,select_cross(list(reversed(bank)),"train1",q))
        self.assertEqual(select_cross(bank,"dev",dict(q,support_indices=[])),[])

    def test_mean_logits_mask_and_two_independent_heads_receive_gradients(self):
        torch.manual_seed(9)
        q=torch.randn(2,256,4,4)
        s=torch.randn(2,3)
        r=torch.randn(2,3,256,4,4)
        mask=torch.tensor([[1.,0,0],[1,1,0]])
        query,relation=QueryHead(),RelationHead()
        pred=relation(q,s,r,mask)
        pred_first=relation(q[:1],s[:1],r[:1,:1],torch.ones(1,1))
        torch.testing.assert_close(pred[:1],pred_first)
        swapped=relation(q,s,r[:,[1,0,2]],mask[:,[1,0,2]])
        torch.testing.assert_close(pred,swapped)
        altered=r.clone();altered[:,2]=999
        torch.testing.assert_close(pred,relation(q,s,altered,mask))
        expected=(relation(q[1:],s[1:],r[1:,:1],torch.ones(1,1))+
                  relation(q[1:],s[1:],r[1:,1:2],torch.ones(1,1)))/2
        torch.testing.assert_close(pred[1:],expected)
        for model,prediction in ((query,query(q,s)),(relation,pred)):
            torch.nn.functional.binary_cross_entropy_with_logits(prediction,torch.tensor([0.,1.])).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertGreater(model.projection.weight.grad.abs().sum().item(),0)
        self.assertEqual(sum(p.numel() for p in query.parameters()),70153)
        self.assertEqual(sum(p.numel() for p in relation.parameters()),68233)
        with self.assertRaises(ValueError):
            relation(q,s,r,torch.zeros_like(mask))

    def test_fixed_sampling_labels_and_minimum_evidence(self):
        labels={(seq,i):i%2 for seq in ("a","b","c") for i in range(40)}
        training=dict(seed=20261023,epochs=10,samples_per_epoch=4000,minimum_negative_rows=50,minimum_negative_sequences=3)
        samples=sample_indices(labels,training)
        self.assertEqual(samples,sample_indices(labels,training))
        self.assertEqual(sum(len(e) for e in samples),40000)
        self.assertTrue(all(label==i%2 for epoch in samples for i,(_,_,label) in enumerate(epoch)))
        with self.assertRaises(ValueError):
            sample_indices({key:value for key,value in labels.items() if key[0]!='c'},training)

    def fixture(self, root):
        cfg=json.loads(CONFIG.read_text())
        cfg['splits'].update(train_numbers=[1,3],calibration_numbers=[],dev_numbers=[])
        cfg['cells']={'test':dict(dataset_root='data',split='train',reference_tracks='tracks',detections='cache',
                                  verifiers={'seed':{'kind':'seed','weights':'seed.pt'}})}
        (root/'config.json').write_text(json.dumps(cfg))
        (root/'seed.pt').write_bytes(b'not a model')
        (root/'tracks').mkdir()
        for seq in ('UAVSwarm-01','UAVSwarm-03'):
            directory=root/'data/train'/seq
            (directory/'img1').mkdir(parents=True)
            (directory/'gt').mkdir();(directory/'gt/gt.txt').write_text('MUST NOT READ')
            (directory/'seqinfo.ini').write_text('[Sequence]\nseqLength=6\nimDir=img1\nimExt=.jpg\n')
            lines=[];detections=[]
            for f in range(1,7):
                (directory/'img1'/f'{f:06d}.jpg').write_bytes(bytes([f]))
                for identity,x in ((1,0),(2,100)):
                    lines.append(f'{f},{identity},{x:.1f},0,10,10,.900,-1,-1,-1\n')
                    detections.append(f'{f},-1,{x},0,10,10,.9,-1,-1,-1\n')
            (root/'tracks'/f'{seq}.txt').write_text(''.join(lines))
            (root/'cache'/seq).mkdir(parents=True)
            (root/'cache'/seq/'det.txt').write_text(''.join(detections))
        return root/'config.json'

    def fake_encoder(self, weights, device):
        def encode(image, boxes):
            n=len(boxes)
            return np.full((n,256,4,4),image[0,0,0],np.float32),np.arange(n,dtype=np.float32),np.ones((n,2),np.float32)
        return encode

    def test_gt_free_extract_identity_feature_alignment_and_strict_apply(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);cfg=self.fixture(root)
            original_open=Path.open
            def guard(path,*args,**kwargs):
                if 'gt' in path.parts:
                    raise AssertionError('GT was opened')
                return original_open(path,*args,**kwargs)
            with patch.object(Path,'open',guard):
                summary=extract(root,cfg,'test',root/'extract','cpu',encoder_loader=self.fake_encoder,
                    image_decoder=lambda b,p:np.full((10,200,3),b[0],np.uint8))
            self.assertFalse(summary['gt_used'])
            _,rows,queries,bodies,_,_=load_extraction(root/'extract',Evidence())
            for seq in rows:
                self.assertEqual((root/'tracks'/f'{seq}.txt').read_bytes(),(root/'extract/tracks/A'/f'{seq}.txt').read_bytes())
                self.assertEqual(bodies[seq].dtype,np.float16)
                self.assertEqual(float(bodies[seq][10,0,0,0]),6.)
                self.assertEqual(queries[seq][10]['support_indices'],[11])
            scoring=root/'scoring';(scoring/'tracks/A').mkdir(parents=True);(scoring/'scores').mkdir()
            for seq,qs in queries.items():
                (scoring/'tracks/A'/f'{seq}.txt').write_bytes(b''.join(r.raw for r in rows[seq]))
                for q in qs:
                    q.update({arm: (0.5 if q['row_index']==10 else 0.4) if q['eligible'] else '' for arm in ARMS})
                    q['cross_refs']=[]
                write_queries(scoring/'scores'/f'{seq}.csv',qs,SCORE_COLUMNS)
            manifest=dict(status='completed',stage='score',sequence_names=list(rows),output_hashes={
                str(p.relative_to(scoring)):sha256(p) for p in scoring.rglob('*') if p.is_file()})
            (scoring/'summary.json').write_text(json.dumps(manifest))
            thresholds=root/'thresholds.json';thresholds.write_text(json.dumps(dict.fromkeys(ARMS,.5)))
            result=apply(scoring,thresholds,root/'applied')
            self.assertEqual(result['retained_rows']['peer'],22)
            for seq in rows:
                expected=b''.join(r.raw for r in rows[seq] if r.row_index!=11)
                self.assertEqual((root/'applied/tracks/A.peer'/f'{seq}.txt').read_bytes(),expected)
                # Exact equality retains; unsupported low-score rows are never removed.
                self.assertIn(rows[seq][10].raw,expected)
            seq='UAVSwarm-01'
            wrong=queries[seq];wrong[0]['x']+=.1
            write_queries(root/'bad.csv',wrong,SCORE_COLUMNS)
            with self.assertRaisesRegex(ValueError,'row identity'):
                read_queries(root/'bad.csv',rows[seq],scoring=True)

    def test_encoder_body_eval_no_grad_and_single_forward(self):
        from fctrack.verifier import SeedVerifier
        model=SeedVerifier().eval()
        before={k:v.clone() for k,v in model.state_dict().items()}
        calls=[]
        hook=model.body.register_forward_hook(lambda *_: calls.append(1))
        with patch('fctrack.verifier.load',return_value=model):
            encode=load_encoder('unused','cpu')
        body,logit,scale=encode(np.full((60,100,3),123,np.uint8),np.array([[20,20,10,10],[40,20,10,10]]))
        hook.remove()
        self.assertEqual(calls,[1]);self.assertEqual(body.shape,(2,256,4,4))
        self.assertEqual(logit.shape,(2,));self.assertEqual(scale.shape,(2,2))
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,before[key])


if __name__=='__main__':
    unittest.main()
