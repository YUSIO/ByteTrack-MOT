"""Independent numerical checks for frozen source attribution."""
import unittest
import numpy as np
import torch
from feedback.trades.model import TraDeSAdapter
from feedback.trades.data import render_heat
from feedback.trades.weak_history_audit import source_partition, native_capture, delete_source, sample, score_frame


class WeakHistoryAuditTest(unittest.TestCase):
    def test_max_heat_partition_and_duplicate_ties(self):
        boxes=np.array([[8,8,32,32],[16,16,40,40],[8,8,32,32]],np.float32)
        parts,ties=source_partition(boxes,1.,(64,64))
        torch.testing.assert_close(parts.sum(0),render_heat(boxes,(64,64))[0],atol=1e-7,rtol=0)
        torch.testing.assert_close(parts[0],parts[2])
        self.assertGreater(ties,0)

    @torch.no_grad()
    def test_deletion_additivity_and_manual_dcn(self):
        torch.manual_seed(8);torch.set_num_threads(2)
        adapter=TraDeSAdapter(channels=8,feature_size=(8,8)).eval()
        adapter.fusion_mode='visual_only'
        torch.nn.init.normal_(adapter.output.weight,std=.1)
        cur=torch.randn(1,8,8,8);prev=torch.randn_like(cur)
        parts,_=source_partition(np.array([[8,8,32,32],[24,24,48,48]],np.float32),1.,(64,64))
        heat=parts.sum(0)[None,None]
        native,_,_,cap=native_capture(adapter,cur,prev,heat)
        drop,own,delta,remaining=delete_source(adapter,cur,cap,parts[0][None,None])
        torch.testing.assert_close(own+remaining,cap['prop'][0],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(native-drop,delta,atol=1e-6,rtol=1e-5)
        all_drop,_,_,_=delete_source(adapter,cur,cap,heat)
        torch.testing.assert_close(all_drop,cur,atol=1e-6,rtol=0)
        kernel=torch.tensor([(x-1,y-1) for y in range(3) for x in range(3)])
        points=torch.tensor([3,4])+kernel+cap['offset'].reshape(1,9,2,8,8)[0,:,:,4,3][:,[1,0]]
        values=sample(cap['support'],points)
        weight=adapter.core.dcn1_1.weight.reshape(64,64,9).permute(2,0,1)
        result=torch.einsum('koi,ki->o',weight,values)+adapter.core.dcn1_1.bias
        torch.testing.assert_close(result,cap['prop'][0][0,:,4,3],atol=1e-6,rtol=1e-5)

    def test_frame_matching_thresholds_and_duplicates(self):
        gt={7:np.array([0,0,10,10]),8:np.array([30,30,10,10])}
        lines='1,-1,0,0,10,10,0.7,-1,-1,-1\n1,-1,0,0,10,10,0.8,-1,-1,-1\n1,-1,30,30,10,10,0.59,-1,-1,-1\n'
        result=score_frame(lines,1,gt,(64,64))
        self.assertEqual(result['50'],dict(TP=1,FN=1,FP=1,matched_ids=[7]))
        self.assertEqual(score_frame('',1,gt,(64,64))['50']['FN'],2)


if __name__=='__main__':unittest.main()
