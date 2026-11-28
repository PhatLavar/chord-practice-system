"""Checks for matched-seed aggregation and descriptive variation."""

import sys
import unittest
import json
import os
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


from acr.experiments.billboard.repeat_frame import METRICS, aggregate, run


class RepeatSeedTests(unittest.TestCase):
    def test_paired_means_and_sample_standard_deviation(self):
        rows = []
        for seed, value in enumerate((0.2, 0.4, 0.6)):
            for method, delta in (("baseline", 0), ("balanced", 0.1)):
                rows.append(dict(method=method, seed=seed, **{k: value + delta for k in METRICS}))
        summary = aggregate(rows)
        self.assertEqual(summary["n"], 3)
        for metric in METRICS:
            self.assertAlmostEqual(summary["methods"]["baseline"][metric]["mean"], 0.4)
            self.assertAlmostEqual(summary["methods"]["baseline"][metric]["sample_std"], 0.2)
            self.assertAlmostEqual(summary["paired_delta"][metric]["mean"], 0.1)
            self.assertAlmostEqual(summary["paired_delta"][metric]["sample_std"], 0)
            self.assertEqual(summary["paired_delta"][metric]["positive_seeds"], 3)

    def test_missing_and_duplicate_pairs_are_rejected(self):
        row = dict(method="baseline", seed=1, **{k: 0.5 for k in METRICS})
        for rows in ([], [row], [row, row]):
            with self.assertRaises(ValueError):
                aggregate(rows)

    def test_invalid_metric_is_rejected(self):
        row = dict(seed=1, **{k: 0.5 for k in METRICS})
        with self.assertRaises(ValueError):
            aggregate(
                [
                    dict(row, method="baseline"),
                    dict(row, method="balanced", frame_accuracy=float("nan")),
                ]
            )

    def test_four_jobs_original_pair_and_completed_run_reuse(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                configs = {
                    "baseline": {"seed": 20261119},
                    "balanced": {"seed": 20261119, "sampling": "quality_song_mixture"},
                }
                Path("configs").mkdir()
                for method, cfg in configs.items():
                    Path(f"configs/{method}.json").write_text(json.dumps(cfg))
                plan = dict(
                    seeds=[20261120, 20261121],
                    baseline_config="configs/baseline.json",
                    balanced_config="configs/balanced.json",
                )
                Path("plan.json").write_text(json.dumps(plan))
                metrics = {k: 0.5 for k in METRICS[:2]}
                diag = {k: 0.1 for k in METRICS[2:]}
                for method, diagname in (
                    ("baseline", "kaggle-diagnostics"),
                    ("balanced", "balanced-diagnostics"),
                ):
                    base = Path(f"reports/p1/kaggle-{method}")
                    base.mkdir(parents=True)
                    (base / "run.json").write_text(json.dumps({"config": configs[method]}))
                    (base / "metrics.json").write_text(json.dumps(metrics))
                    target = Path(f"reports/p1/{diagname}")
                    target.mkdir()
                    (target / "summary.json").write_text(json.dumps(diag))

                def execute(args, check):
                    target = Path(args[args.index("--output") + 1])
                    target.mkdir(parents=True)
                    if "--config" in args:
                        cfg = json.loads(Path(args[args.index("--config") + 1]).read_text())
                        (target / "run.json").write_text(
                            json.dumps({"config": cfg, "git_revision": "test-revision"})
                        )
                        (target / "metrics.json").write_text(json.dumps(metrics))
                    else:
                        (target / "summary.json").write_text(json.dumps(diag))

                with (
                    patch(
                        "acr.experiments.billboard.repeat_frame.subprocess.check_output",
                        return_value="test-revision",
                    ),
                    patch(
                        "acr.experiments.billboard.repeat_frame.subprocess.run", side_effect=execute
                    ) as launch,
                    redirect_stdout(StringIO()),
                ):
                    run(Path("plan.json"), Path("outputs"), "cpu")
                    self.assertEqual(launch.call_count, 8)
                    self.assertEqual(len(json.loads(Path("outputs/completed.json").read_text())), 6)
                    run(Path("plan.json"), Path("outputs"), "cpu")
                    self.assertEqual(launch.call_count, 8)
                summary = json.loads(Path("outputs/summary.json").read_text())
                self.assertEqual(summary["seeds"], [20261119, 20261120, 20261121])
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
