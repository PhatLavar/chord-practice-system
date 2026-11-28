"""Padding, frame coverage and gradient checks for the Transformer teacher."""

import sys, unittest
from pathlib import Path

import numpy as np
import torch
from acr.models.transformer import TransformerFrameTeacher
from acr.models.components import losses
from acr.data.windows import Windows
from acr.training.frame import recording_logits


class TransformerTests(unittest.TestCase):
    def test_padding_is_not_silence_and_does_not_change_valid_outputs(self):
        torch.manual_seed(1)
        model = TransformerFrameTeacher(window=4).eval()
        song = {"x": np.zeros((3, 24), dtype=np.float32), "y": np.array([1, 0, -100])}
        dataset = Windows({"song": song}, 4, 31, include_padding_mask=True)
        x, y, padding = dataset[0]
        self.assertFalse(padding[31:34].any())
        self.assertTrue(padding[:31].all())
        first = model(x.unsqueeze(0), padding.unsqueeze(0))
        changed = x.clone()
        changed[:, padding] = 999
        second = model(changed.unsqueeze(0), padding.unsqueeze(0))
        for a, b in zip(first, second):
            torch.testing.assert_close(a[:, :, 31:34], b[:, :, 31:34])
        terms = losses([h[:, :, 31:35] for h in first], y.unsqueeze(0))
        sum(terms).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_inference_keeps_unlabeled_chunks_and_tail(self):
        model = TransformerFrameTeacher(window=4).eval()
        song = {"x": np.zeros((9, 24), dtype=np.float32), "y": np.full(9, -100)}
        heads = recording_logits(model, "song", song, "cpu")
        self.assertEqual([h.shape for h in heads], [(1, 2, 9), (1, 12, 9), (1, 6, 9), (1, 5, 9)])
        self.assertTrue(all(torch.isfinite(h).all() for h in heads))

    def test_all_padding_is_rejected(self):
        model = TransformerFrameTeacher(window=4)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 24, 66), torch.ones(1, 66, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
