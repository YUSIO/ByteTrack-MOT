import tempfile,unittest
from pathlib import Path
import cv2,numpy as np,torch
from fctrack.temporal_birth import adjacent_links,ClipBuilder,TemporalBirth,augment
class TemporalBirthTests(unittest.TestCase):
    def test_matching_one_to_one_deterministic_and_empty(self):
        prev=np.array([[10,10,10,10,.2],[50,10,10,10,.2]],float)
        cur=np.array([[11,10,10,10,.2],[12,10,10,10,.2],[51,10,10,10,.2]])
        np.testing.assert_array_equal(adjacent_links(prev,cur),[0,-1,1]);self.assertEqual(adjacent_links(np.empty((0,5)),cur).tolist(),[-1]*3)
    def test_actual_clip_history_masks_and_no_future(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);(d/'img1').mkdir()
            for f in [1,2,3]:cv2.imwrite(str(d/'img1'/f'{f:06d}.jpg'),np.full((40,40,3),f*30,np.uint8))
            cache=np.array([[f,-1,10,10,10,10,.2,-1,-1,-1] for f in [1,2,3]],float)
            builder=ClipBuilder(d,cache);clip,meta=builder.clip(1,0)
            self.assertEqual(clip.shape,(3,3,64,64));self.assertEqual(meta[[6,14,22]].tolist(),[0,0,1])
            self.assertFalse(clip[:2].any());clip,meta=builder.clip(3,0)
            self.assertEqual(meta[[5,13,21]].tolist(),[1,1,1]);self.assertLess(clip[0].mean(),clip[2].mean())
            (d/'img1/000003.jpg').unlink();builder=ClipBuilder(d,cache);builder.clip(2,0)
    def test_missing_detection_keeps_real_past_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);(d/'img1').mkdir()
            for f in [1,2]:cv2.imwrite(str(d/'img1'/f'{f:06d}.jpg'),np.full((40,40,3),60,np.uint8))
            builder=ClipBuilder(d,np.array([[2,-1,10,10,10,10,.2,-1,-1,-1]],float));clip,meta=builder.clip(2,0)
            self.assertEqual(meta[13],0);self.assertEqual(meta[14],1);self.assertTrue(clip[1].any())
    def test_architecture_and_BCE_gradient_finite(self):
        torch.set_num_threads(1);model=TemporalBirth();model.train();model.freeze_bn_statistics()
        clips=torch.rand(2,3,3,64,64);metadata=torch.zeros(2,26);metadata[:,[6,14,22]]=1
        logits=model(clips,metadata);self.assertEqual(tuple(logits.shape),(2,))
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([0.,1.]));loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        model.eval();a=model(clips,metadata,'S1');clips[:,:2]=torch.rand_like(clips[:,:2]);b=model(clips,metadata,'S1');torch.testing.assert_close(a,b)
    def test_augmentation_keeps_missing_zero_and_shared_transform(self):
        x=torch.ones(4,3,3,64,64)*.5;x[:,0]=0;meta=torch.ones(4,26)
        y,m=augment(x,meta,torch.Generator().manual_seed(5));self.assertFalse(y[:,0].any());torch.testing.assert_close(y[:,1],y[:,2]);self.assertTrue(torch.equal(m[:,0],m[:,8]))
if __name__=='__main__':unittest.main()
