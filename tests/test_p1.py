"""Regression checks for target masks, decoder and sequence-window coverage."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src/acr"))
import numpy as np
import torch
from p1_data import encode, CLASSES
from teacher_f import targets, losses, decode
from train_f import Windows


class P1Tests(unittest.TestCase):
    def test_all_supported_chords_decode_from_components(self):
        y=torch.arange(len(CLASSES)).unsqueeze(0)
        heads=[]
        for target,size in zip(targets(y),(2,12,6,5)):
            logits=torch.full((1,size,len(CLASSES)),-20.)
            logits.scatter_(1,target.clamp_min(0).unsqueeze(1),20.)
            heads.append(logits)
        self.assertTrue(torch.equal(decode(heads),y))

    def test_unknown_and_no_chord_masks(self):
        self.assertEqual(encode("C:hdim7"),encode("B#:hdim7/b3"))
        self.assertEqual(encode("C:9"),-100)
        self.assertEqual(encode("X"),-100)
        self.assertEqual(encode(""),-100)
        labels=torch.tensor([[-100,0]])
        for head in targets(labels)[1:]:
            self.assertTrue((head==-100).all())
        logits=[torch.randn(1,n,2,requires_grad=True) for n in (2,12,6,5)]
        value=sum(losses(logits,torch.full((1,2),-100)))
        self.assertEqual(value.item(),0)
        value.backward()

    def test_tail_padding_and_context(self):
        x=np.arange(24*9,dtype=np.float32).reshape(9,24)
        y=np.arange(9,dtype=np.int64)
        windows=Windows({"a":{"x":x,"y":y}},4,2)
        recovered=torch.cat([windows[i][1] for i in range(len(windows))])
        self.assertTrue(torch.equal(recovered[:9],torch.arange(9)))
        self.assertTrue((recovered[9:]==-100).all())
        features,_=windows[1]
        np.testing.assert_array_equal(features[:,:7],x[2:9].T)


if __name__=="__main__":
    unittest.main()
