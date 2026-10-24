import tempfile
import unittest
from pathlib import Path
from feedback.validate import evaluate

class ValidationTests(unittest.TestCase):
    def test_perfect_and_empty_predictions_include_empty_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'data';cache=Path(tmp)/'cache';seq='UAVSwarm-01'
            (root/seq/'gt').mkdir(parents=True);(cache/seq).mkdir(parents=True)
            (root/seq/'seqinfo.ini').write_text('[Sequence]\nimWidth=100\nimHeight=100\nseqLength=2\n')
            (root/seq/'gt/gt.txt').write_text('1,1,10,10,20,20,1,1,1\n')
            (cache/seq/'det.txt').write_text('1,-1,10,10,20,20,0.9,-1,-1,-1\n')
            result=evaluate(cache,root,[seq])
            self.assertEqual(result['frames'],2)
            self.assertAlmostEqual(result['metrics']['AP50:95'],1)
            (cache/seq/'det.txt').write_text('')
            self.assertEqual(evaluate(cache,root,[seq])['metrics']['AP50:95'],0)
