"""Auxiliary labels, mixed input branches and fixed-budget experiment controls."""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch

from acr.data.jaah import external_label, parse_intervals, normalize_title
from acr.vocabulary import encode
from acr.data.events import Events, collate_events
from acr.models.tcn import EventTeacher
from acr.models.components import losses
from acr.experiments.jaah.event import MultiInputEventTeacher, replacement_steps


class JaahTests(unittest.TestCase):
    def test_bare_major_only_and_unknowns_stay_masked(self):
        self.assertEqual(external_label("F#"), encode("F#:maj"))
        self.assertEqual(external_label("N"), 0)
        for label in ["X", "C:9", "C:7(b9)", "C:(3,#5,b7)"]:
            self.assertEqual(external_label(label), -100)
        intervals = parse_intervals("0 1 C\n1 2 C:maj\n2 3 X\n3 4 C:maj")
        self.assertEqual(len(intervals), 3)
        self.assertEqual(intervals[0], (0.0, 2.0, encode("C:maj")))
        with self.assertRaises(ValueError):
            parse_intervals("0 2 C\n1 3 D")
        with self.assertRaises(ValueError):
            parse_intervals("0 nan C")

    def test_original_branch_and_rng_unchanged(self):
        torch.manual_seed(17)
        original = EventTeacher()
        state = torch.get_rng_state()
        torch.manual_seed(17)
        mixed = MultiInputEventTeacher()
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        x = torch.rand(2, 24, 70)
        mask = torch.ones(2, 70, dtype=torch.bool)
        for a, b in zip(original(x, mask), mixed(x, mask)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_aux_gradients_reach_shared_heads_not_24_projection(self):
        model = MultiInputEventTeacher()
        x = torch.rand(2, 12, 70)
        mask = torch.ones(2, 70, dtype=torch.bool)
        sum(losses(model(x, mask), torch.tensor([[1], [2]]))).backward()
        self.assertIsNone(model.input.weight.grad)
        for parameter in [model.input12.weight, model.blocks[0].weight, model.heads[1].weight]:
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_12_channel_padding_and_mixed_batch_rejection(self):
        songs = {"a": {"x": np.ones((30, 12), dtype=np.float32)}}
        dataset = Events(songs, [("a", 0.0, 1.0, 0, 10, 1)])
        item = dataset[0]
        x, mask, y = collate_events([item])
        self.assertEqual(x.shape, (1, 12, 72))
        self.assertEqual(mask.sum().item(), 10)
        with self.assertRaises(ValueError):
            collate_events([item, (torch.zeros(24, 72), item[1], 1)])

    def test_fixed_replacement_budget_and_conservative_title(self):
        self.assertEqual(replacement_steps(990, 0, 19, 1), set())
        self.assertEqual(len(replacement_steps(990, 0.25, 19, 1)), 248)
        self.assertEqual(replacement_steps(990, 0.25, 19, 1), replacement_steps(990, 0.25, 19, 1))
        self.assertEqual(
            normalize_title("My Funny Valentine!"), normalize_title("my funny valentine")
        )


if __name__ == "__main__":
    unittest.main()
