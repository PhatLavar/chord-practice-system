"""Regression checks for target masks, decoder and sequence-window coverage."""

import sys
import unittest
from pathlib import Path


import numpy as np
import torch
from acr.vocabulary import encode, CLASSES
from acr.models.components import targets, losses, decode
from acr.data.windows import Windows, QualityBalancedWindows
from acr.training.frame import component_weights
from acr.evaluation.frame import probabilities, decode_components
from acr.data.billboard.audit import reference_events


class P1Tests(unittest.TestCase):
    def test_all_supported_chords_decode_from_components(self):
        y = torch.arange(len(CLASSES)).unsqueeze(0)
        heads = []
        for target, size in zip(targets(y), (2, 12, 6, 5)):
            logits = torch.full((1, size, len(CLASSES)), -20.0)
            logits.scatter_(1, target.clamp_min(0).unsqueeze(1), 20.0)
            heads.append(logits)
        self.assertTrue(torch.equal(decode(heads), y))

    def test_unknown_and_no_chord_masks(self):
        self.assertEqual(encode("C:hdim7"), encode("B#:hdim7/b3"))
        self.assertEqual(encode("C:9"), -100)
        self.assertEqual(encode("X"), -100)
        self.assertEqual(encode(""), -100)
        labels = torch.tensor([[-100, 0]])
        for head in targets(labels)[1:]:
            self.assertTrue((head == -100).all())
        logits = [torch.randn(1, n, 2, requires_grad=True) for n in (2, 12, 6, 5)]
        value = sum(losses(logits, torch.full((1, 2), -100)))
        self.assertEqual(value.item(), 0)
        value.backward()

    def test_reference_events_preserve_unknown_and_merge_inversions(self):
        result = reference_events(
            ["0 1 C:maj", "1 2 C:maj/3", "2 3 X", "3 4 C:maj", "4 5 C:9", "5 6 C:maj"]
        )
        self.assertEqual(
            result,
            [
                (0.0, 2.0, encode("C:maj")),
                (2.0, 3.0, -100),
                (3.0, 4.0, encode("C:maj")),
                (4.0, 5.0, -100),
                (5.0, 6.0, encode("C:maj")),
            ],
        )

    def test_numpy_decoder_matches_teacher_and_probability_pooling(self):
        generator = torch.Generator().manual_seed(7)
        logits = [torch.randn(1, n, 21, generator=generator) for n in (2, 12, 6, 5)]
        probs = [probabilities(h[0].numpy()) for h in logits]
        np.testing.assert_array_equal(decode_components(probs), decode(logits)[0].numpy())
        pooled = [p.mean(axis=1, keepdims=True) for p in probs]
        pooled_logits = [torch.from_numpy(np.log(p)).unsqueeze(0) for p in pooled]
        np.testing.assert_array_equal(decode_components(pooled), decode(pooled_logits)[0].numpy())
        for p in probs:
            np.testing.assert_allclose(p.sum(axis=0), 1.0)

    def test_weight_counts_and_masked_weighted_loss(self):
        labels = np.array([encode("C:maj")] * 100 + [encode("D:dim7"), 0, -100])
        weights, record = component_weights({"train": {"y": labels}}, "cpu")
        self.assertEqual(sum(record["triad"]["counts"]), 101)
        self.assertEqual(sum(record["extension"]["counts"]), 101)
        self.assertIsNone(weights[0])
        self.assertIsNone(weights[1])
        for head in weights[2:]:
            positive = head[head > 0]
            self.assertLessEqual(float(positive.max() / positive.min()), 10.00001)
        y = torch.from_numpy(labels).unsqueeze(0)
        logits = [torch.randn(1, n, len(labels), requires_grad=True) for n in (2, 12, 6, 5)]
        terms = losses(logits, y, weights)
        self.assertTrue(all(torch.isfinite(v) for v in terms))
        # Ignored/N targets cannot change component losses.
        changed = [h.detach().clone() for h in logits]
        for h in changed[1:]:
            h[:, :, -2:] = 999.0
        for a, b in zip(terms[1:], losses(changed, y, weights)[1:]):
            self.assertTrue(torch.allclose(a, b))
        sum(terms).backward()
        for h in logits:
            self.assertTrue(torch.isfinite(h.grad).all())

    def test_balanced_sampling_keeps_rare_anchor_and_budget(self):
        y = np.full(1024, encode("C:maj"), dtype=np.int64)
        y[700] = encode("C:dim7")
        songs = {"train": {"x": np.zeros((1024, 24), dtype=np.float32), "y": y}}
        dataset = QualityBalancedWindows(songs, 512, 31)
        self.assertEqual(len(dataset), len(Windows(songs, 512, 31)))
        self.assertEqual(set(dataset.qualities), {0, 13})
        torch.manual_seed(5)
        first = [dataset[i % len(dataset)][1].clone() for i in range(100)]
        self.assertTrue(any((v == encode("C:dim7")).any() for v in first))
        torch.manual_seed(5)
        second = [dataset[i % len(dataset)][1] for i in range(100)]
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        short = {
            "short": {"x": np.zeros((1, 24), dtype=np.float32), "y": np.array([encode("D:hdim7")])}
        }
        short_dataset = QualityBalancedWindows(short, 512, 31)
        _, target = short_dataset[0]
        self.assertEqual(target[0], encode("D:hdim7"))
        self.assertTrue((target[1:] == -100).all())

    def test_tail_padding_and_context(self):
        x = np.arange(24 * 9, dtype=np.float32).reshape(9, 24)
        y = np.arange(9, dtype=np.int64)
        windows = Windows({"a": {"x": x, "y": y}}, 4, 2)
        recovered = torch.cat([windows[i][1] for i in range(len(windows))])
        self.assertTrue(torch.equal(recovered[:9], torch.arange(9)))
        self.assertTrue((recovered[9:] == -100).all())
        features, _ = windows[1]
        np.testing.assert_array_equal(features[:, :7], x[2:9].T)


if __name__ == "__main__":
    unittest.main()
