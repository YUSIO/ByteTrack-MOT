import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from feedback.validate import evaluate, validate

class ValidationTests(unittest.TestCase):
    def test_interventions_restore_mode_and_use_only_previous_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'data';seq='UAVSwarm-01';path=root/'train'/seq
            (path/'gt').mkdir(parents=True)
            (path/'seqinfo.ini').write_text('[Sequence]\nimWidth=100\nimHeight=100\nseqLength=2\nframeRate=30\n')
            (path/'gt/gt.txt').write_text('1,1,10,10,20,20,1,1,1\n2,2,30,30,20,20,1,1,1\n')
            model=torch.nn.Module();model.mode='pp';model.feedback=torch.nn.Identity()
            seen=[]
            class Runner:
                def __init__(self, model, *args):
                    self.prompts={};self.mode=model.mode
                def step(self, *args):
                    seen.append((self.mode,sorted(self.prompts)))
                    self.prompts={99:np.array([0,0,1,1])}
                    return {'detections':np.empty((0,5)),'tracks':[]}
            with patch('feedback.validate.OnlineFeedback',Runner),patch('feedback.validate.read_image',return_value=(torch.zeros(3,32,32),1.,(100,100))):
                validate(model,{"input_size":[32,32]},None,root,[seq],Path(tmp)/'gt','cpu','gt_history')
                self.assertEqual(seen,[('pp',[]),('pp',[1])])
                seen.clear()
                validate(model,{"input_size":[32,32]},None,root,[seq],Path(tmp)/'off','cpu','off')
                self.assertEqual(seen[0][0],'none');self.assertEqual(model.mode,'pp')
                seen.clear()
                validate(model,{"input_size":[32,32]},None,root,[seq],Path(tmp)/'online','cpu','online')
                self.assertEqual(seen,[('pp',[]),('pp',[99])])

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
