import tempfile
import unittest
from pathlib import Path
from feedback.trades.train import check_config
from feedback.trades.validate import evaluate


class PilotTests(unittest.TestCase):
    def test_split_and_execution_constraints(self):
        c=dict(fit_sequences=['UAVSwarm-05'],dev_sequences=['UAVSwarm-01'],quarantine_sequences=['UAVSwarm-03'],
               epochs=1,batch_size=1,precision='fp32',test_access=False,resume=False,gradient_clip=None)
        check_config(c)
        for updates in [dict(dev_sequences=['UAVSwarm-05']),dict(test_access=True),dict(epochs=2),
                        dict(fit_sequences=['UAVSwarm-03']),dict(gradient_clip=1)]:
            with self.assertRaises(ValueError):check_config(dict(c,**updates))

    def test_evaluation_counts_empty_gt_frames_and_empty_detections(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);seq=root/'train/UAVSwarm-01';(seq/'gt').mkdir(parents=True)
            (seq/'seqinfo.ini').write_text('[Sequence]\nimwidth=100\nimheight=100\nseqlength=3\n')
            (seq/'gt/gt.txt').write_text('1,1,10,10,20,20,1,1,1\n')
            cache=root/'cache';out=cache/'UAVSwarm-01';out.mkdir(parents=True)
            (out/'det.txt').write_text('1,-1,10,10,20,20,0.9,-1,-1,-1\n2,-1,50,50,10,10,0.7,-1,-1,-1\n')
            result=evaluate(cache,root,['UAVSwarm-01'])
            self.assertEqual(result['frames'],3)
            self.assertEqual(result['fixed_iou50_score06'],dict(TP=1,FP=1,FN=0,recall=1))
            (out/'det.txt').write_text('')
            result=evaluate(cache,root,['UAVSwarm-01'])
            self.assertEqual(result['fixed_iou50_score06'],dict(TP=0,FP=0,FN=1,recall=0))


if __name__=='__main__':unittest.main()
