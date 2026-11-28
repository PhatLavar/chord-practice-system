"""Oracle event pooling, padding and F/E comparison checks."""

import sys, unittest, tempfile, csv, tarfile, hashlib, json, io
from pathlib import Path

import numpy as np
import torch
from unittest.mock import patch
from acr.data.events import Events, collate_events, load_events
from acr.models.tcn import EventTeacher
from acr.models.tcn import FrameTeacher
from acr.models.components import losses
from acr.training.event import compare, FIELDS, evaluate


class EventTests(unittest.TestCase):
    def test_masked_mean_excludes_context_and_padding(self):
        model = EventTeacher(width=24)
        model.encode_features = lambda x: x
        x = torch.arange(24 * 8, dtype=torch.float32).reshape(1, 24, 8)
        mask = torch.tensor([[False, False, True, True, False, False, False, False]])
        actual = model(x, mask)
        for head, out in zip(model.heads, actual):
            torch.testing.assert_close(out, head(x[:, :, 2:4].mean(2, keepdim=True)))
        with self.assertRaises(ValueError):
            model(x, torch.zeros_like(mask))

    def test_variable_event_batch_and_gradients(self):
        song = {"x": np.random.default_rng(2).random((100, 24), dtype=np.float32)}
        events = [("a", 0.0, 1.0, 0, 1, 1), ("a", 2.0, 5.0, 40, 60, 0)]
        dataset = Events({"a": song}, events)
        batch = collate_events([dataset[0], dataset[1]])
        self.assertEqual(batch[1].sum(1).tolist(), [1, 20])
        model = EventTeacher()
        heads = model(batch[0], batch[1])
        model.eval()
        separate = model(dataset[0][0].unsqueeze(0), dataset[0][1].unsqueeze(0))
        for first, combined in zip(separate, heads):
            torch.testing.assert_close(first[0], combined[0], atol=1e-5, rtol=1e-5)
        loss = sum(losses(heads, batch[2]))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_event_loading_respects_times_unknowns_and_empty_events(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            labels = p / "labels.tar"
            text = b"0 0.5 C:maj\n0.5 1 C:min\n1 1.01 C:7\n1.01 2 X\n"
            with tarfile.open(labels, "w") as archive:
                member = tarfile.TarInfo("McGill-Billboard/0001/full.lab")
                member.size = len(text)
                archive.addfile(member, io.BytesIO(text))
            (p / "metadata.json").write_text(
                json.dumps(
                    {
                        "source_sha256": {
                            "full.tar.xz": hashlib.sha256(labels.read_bytes()).hexdigest()
                        }
                    }
                )
            )
            np.savez(
                p / "0001.npz",
                x=np.zeros((3, 24), dtype=np.float32),
                y=np.array([1, 2, -100]),
                times=np.array([0.0, 0.5, 1.1]),
            )
            _, events, empty = load_events(p, labels, ["0001"])
            self.assertEqual(empty, 1)
            self.assertEqual([(e[3], e[4], e[5]) for e in events], [(0, 1, 1), (1, 2, 2)])

    def test_event_sampling_is_reproducible_and_budget_is_unchanged(self):
        song = {"x": np.zeros((100, 24), dtype=np.float32)}
        events = [("a", 0.0, 1.0, 0, 10, 1), ("a", 1.0, 2.0, 10, 11, 14)]
        dataset = Events({"a": song}, events, balanced=True)
        self.assertEqual(len(dataset), 2)
        torch.manual_seed(3)
        first = [dataset[i % 2][2] for i in range(50)]
        torch.manual_seed(3)
        self.assertEqual(first, [dataset[i % 2][2] for i in range(50)])
        self.assertEqual(set(first), {1, 14})

    def test_original_half_mixture_sequence_is_preserved(self):
        song = {"x": np.zeros((100, 24), dtype=np.float32)}
        events = [("a", 0.0, 1.0, 0, 10, 1), ("a", 1.0, 2.0, 10, 11, 14)]
        dataset = Events({"a": song}, events, balanced=True)
        torch.manual_seed(3)
        self.assertEqual(
            [dataset[i % 2][2] for i in range(12)], [1, 14, 1, 14, 1, 14, 14, 14, 14, 14, 1, 1]
        )

    def test_quarter_mixture_threshold_and_endpoint(self):
        song = {"x": np.zeros((100, 24), dtype=np.float32)}
        events = [("a", 0.0, 1.0, 0, 10, 1), ("a", 1.0, 2.0, 10, 11, 14)]
        dataset = Events({"a": song}, events, balanced=True, balanced_probability=0.25)
        with (
            patch("acr.data.events.torch.rand", return_value=torch.tensor(0.74)),
            patch("acr.data.events.torch.randint") as draw,
        ):
            self.assertEqual(dataset[0][2], 1)
            draw.assert_not_called()
        with (
            patch("acr.data.events.torch.rand", return_value=torch.tensor(0.75)),
            patch(
                "acr.data.events.torch.randint",
                side_effect=[torch.tensor(1), torch.tensor(0), torch.tensor(0)],
            ),
        ):
            self.assertEqual(dataset[0][2], 14)
        natural = Events({"a": song}, events, balanced=True, balanced_probability=0)
        self.assertEqual([natural[i % 2][2] for i in range(12)], [1, 14] * 6)

    def test_invalid_mixture_probability_is_rejected(self):
        for value in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                Events({}, [], balanced_probability=value)

    def test_frame_teacher_state_keys_are_preserved(self):
        torch.manual_seed(9)
        f = FrameTeacher()
        x = torch.randn(1, 24, 100)
        h = torch.relu(f.input(x))
        for block in f.blocks:
            h = h + torch.relu(block(h))
        for expected, actual in zip([head(h) for head in f.heads], f(x)):
            torch.testing.assert_close(expected, actual)
        e = EventTeacher()
        self.assertEqual(set(f.state_dict()), set(e.state_dict()))

    def test_complementarity_and_masks(self):
        with tempfile.TemporaryDirectory() as name:
            p = Path(name)
            # One F-only C:maj, one E-only N. N's chord components are masked.
            f_rows = [["a", 0, 1, 1, 1, 0, 0, 0, 0], ["a", 1, 2, 0, 1, 0, 0, 0, 0]]
            e_rows = [["a", 0, 1, 1, 15, 0, 1, 0, 0], ["a", 1, 2, 0, 0, 1, 0, 0, 0]]
            with (p / "f.csv").open("w", newline="") as stream:
                w = csv.writer(stream)
                w.writerow(FIELDS)
                w.writerows(f_rows)
            compare(p / "f.csv", e_rows, p)
            with (p / "complementarity.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            lookup = {(r["quality"], r["metric"]): r for r in rows}
            self.assertEqual(lookup["maj", "complete"]["F_only"], "1")
            self.assertEqual(lookup["N", "complete"]["E_only"], "1")
            self.assertNotIn(("N", "root"), lookup)
            with self.assertRaises(ValueError):
                compare(p / "f.csv", e_rows[:1], p)


if __name__ == "__main__":
    unittest.main()
