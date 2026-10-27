import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from feedback.trades.continue_training import restore_optimizer,resume_order,validate_resume_config


class ResumeTests(unittest.TestCase):
    def test_momentum_and_rng_resume_matches_continuous_update(self):
        torch.manual_seed(13)
        config=dict(seed=0,lr=1e-4,momentum=.9,weight_decay=1e-4)
        model=torch.nn.Linear(3,2)
        optimizer=torch.optim.SGD(model.parameters(),lr=config['lr'],momentum=.9,weight_decay=1e-4)
        def step(m,o):
            o.zero_grad();loss=m(torch.randn(4,3)).square().sum();loss.backward();o.step()
        step(model,optimizer);weights=copy.deepcopy(model.state_dict())
        snapshot=dict(epoch=1,seed=0,optimizer=copy.deepcopy(optimizer.state_dict()),
                      torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        step(model,optimizer);expected=copy.deepcopy(model.state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'optimizer.pt';torch.save(snapshot,path)
            resumed=torch.nn.Linear(3,2);resumed.load_state_dict(weights)
            other=torch.optim.SGD(resumed.parameters(),lr=config['lr'],momentum=.9,weight_decay=1e-4)
            restore_optimizer(other,path,config);step(resumed,other)
            for k,value in resumed.state_dict().items():self.assertTrue(torch.equal(value,expected[k]))
            self.assertEqual(len(other.state),2)
            with self.assertRaises(ValueError):restore_optimizer(other,path,dict(config,lr=.01))

    def test_order_rebuild_continues_stream_without_repeating_epoch1(self):
        reference=np.random.default_rng(0);first=reference.permutation(20).tolist();second=reference.permutation(20).tolist()
        recovered=resume_order(0,20,[dict(index=i) for i in first])
        self.assertEqual(recovered.permutation(20).tolist(),second)
        with self.assertRaises(ValueError):resume_order(0,20,[dict(index=i) for i in reversed(first)])

    def test_resume_rejects_changed_training_and_data_settings(self):
        keys=['batch_size','precision','seed','input_size','optimizer','lr','momentum','weight_decay',
              'cva_weight','gradient_clip','expected_fit_pairs','validation_modes','base_sha256',
              'history','dev_history','heat_shift','detector_seen_dev','inventories',
              'fit_sequences','dev_sequences','quarantine_sequences','gpu_uuid']
        parent={k:k for k in keys};current=dict(parent,epochs=10,start_epoch=2,resume=True,test_access=False)
        validate_resume_config(current,parent)
        for updates in [dict(lr='changed'),dict(fit_sequences=[]),dict(cva_weight=1),dict(epochs=11),dict(test_access=True)]:
            with self.assertRaises(ValueError):validate_resume_config(dict(current,**updates),parent)


if __name__=='__main__':unittest.main()
