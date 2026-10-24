import tempfile
import unittest
from pathlib import Path
import torch
from feedback.data import TrajectoryWindows


class DataTests(unittest.TestCase):
    def test_synchronized_window_and_uncertainty_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);seq=root/'train'/'sample';(seq/'gt').mkdir(parents=True)
            (seq/'seqinfo.ini').write_text('[Sequence]\nseqLength=20\nimWidth=100\nimHeight=100\nframeRate=30\n')
            rows=[]
            for t in range(1,21):
                rows.append(f'{t},1,{t*t},0,2,2,1,1,1\n')
                if t!=10:rows.append(f'{t},2,{t},0,2,2,1,1,1\n')
            (seq/'gt/gt.txt').write_text(''.join(rows))
            data=TrajectoryWindows(root,['sample'],[100,100]);self.assertEqual(len(data),1)
            sample=data[0]
            self.assertEqual(sample['history'].shape,(1,8,2))
            self.assertEqual(sample['future'].shape,(1,12,2))
            self.assertEqual(sample['history'][0,-1,0].item(),65)
            self.assertEqual(sample['future'][0,0,0].item(),82)
            self.assertAlmostEqual(sample['reference'][0,0].item(),sum(t*t+1 for t in range(6,13))/7)
            with self.assertRaises(ValueError):TrajectoryWindows(root,['../test/sample'],[100,100])


if __name__=='__main__':unittest.main()
