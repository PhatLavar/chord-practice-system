"""Frame-input parity, probability pooling and shared-error pair diagnostics."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/acr'))
from teacher_f import FrameTeacher, targets, losses
from experiment_jaah import MultiInputFrameTeacher
from train_f import QualityBalancedWindows
from experiment_jaah_f import pool_events
from compare_jaah_pairs import load_result, summarize_pair, complete_scores
from diagnose_f import probabilities


class JaahPairTests(unittest.TestCase):
    def test_frame_branch_parity_and_auxiliary_gradients(self):
        torch.manual_seed(3);original=FrameTeacher();state=torch.get_rng_state()
        torch.manual_seed(3);model=MultiInputFrameTeacher()
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        x=torch.rand(2,24,74)
        for a,b in zip(original(x),model(x)):torch.testing.assert_close(a,b,rtol=0,atol=0)
        logits=model(torch.rand(2,12,74))
        sum(losses(logits,torch.ones(2,74,dtype=torch.long))).backward()
        self.assertIsNone(model.input.weight.grad)
        self.assertGreater(model.input12.weight.grad.abs().sum().item(),0)
        self.assertGreater(model.blocks[0].weight.grad.abs().sum().item(),0)

    def test_frame_windows_support_12_bins_and_keep_masks(self):
        songs={'j':dict(x=np.ones((8,12),dtype=np.float32),y=np.array([1,1,-100,-100,2,2,0,0]))}
        dataset=QualityBalancedWindows(songs,10,2)
        x,y=dataset.get_window('j',0)
        self.assertEqual(tuple(x.shape),(12,14));self.assertEqual(y.tolist(),[1,1,-100,-100,2,2,0,0,-100,-100])
        self.assertTrue((x[:,:2]==0).all())

    def test_pooling_uses_mean_probabilities_and_checks_saved_labels(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root);out=p/'out';out.mkdir()
            logits=[np.zeros((n,3)) for n in (2,12,6,5)]
            logits[1][0]=[10,-2,-2];logits[1][1]=[0,2,2]
            np.savez(p/'a.npz',times=np.array([0.,1.,2.]),reference=np.ones(3,dtype=int),
                     **{f'logits_{j}':v for j,v in enumerate(logits)})
            pool_events(p,[('a',0.,3.,0,3,1)],out)
            with np.load(out/'event_probabilities.npz') as data:
                actual=data['probabilities_1']
            np.testing.assert_allclose(actual,probabilities(logits[1]).mean(1,keepdims=True))
            self.assertFalse(np.allclose(actual,probabilities(logits[1].mean(1,keepdims=True))))
            (out/'run.json').write_text('{}')
            load_result(out,'F')
            with self.assertRaises(ValueError):pool_events(p,[('a',0.,3.,0,3,2)],out)
            (out/'events.csv').write_text((out/'events.csv').read_text().replace(',1,',',2,',1))
            with self.assertRaises(ValueError):load_result(out,'F')

    def test_agreement_can_be_confident_and_wrong_on_rare_chord(self):
        y=np.array([3]);wrong=targets(torch.tensor([1]))
        probs=[]
        for size,target in zip((2,12,6,5),wrong):
            p=np.zeros((size,1));p[int(target[0]),0]=1;probs.append(p)
        result=summarize_pair(probs,probs,y,np.array(['song']),['dim'])
        self.assertEqual(result['true_rare']['both_wrong'],1)
        self.assertEqual(result['true_rare']['complete_agreement']['coverage'],1)
        self.assertEqual(result['true_rare']['complete_agreement']['precision'],0)
        metrics,rows=complete_scores(y,np.array([1]),['dim'])
        self.assertEqual(metrics['rare_macro_event_recall'],0)
        self.assertEqual(next(r for r in rows if r['quality']=='maj')['precision'],0)


if __name__=='__main__':unittest.main()
