"""Compare E's 0%, 25% and 50% sampling policies on matched seeds."""

import argparse
import csv
import json
from pathlib import Path

from acr.experiments.billboard.repeat_event import METRICS, read_result
from acr.experiments.billboard.repeat_frame import aggregate


def summarize(rows):
    methods = ("natural", "mixture25", "balanced")
    seeds = sorted({row["seed"] for row in rows})
    if len(rows) != len(seeds) * 3 or any(
        {r["method"] for r in rows if r["seed"] == seed} != set(methods) for seed in seeds
    ):
        raise ValueError("Every seed needs one result per sampling policy")
    pairs = {}
    for reference in ("natural", "balanced"):
        pair = [r for r in rows if r["method"] in (reference, "mixture25")]
        pairs[reference] = aggregate(pair, metrics=METRICS, methods=(reference, "mixture25"))
    return dict(
        seeds=seeds,
        n=len(seeds),
        methods=dict(
            natural=pairs["natural"]["methods"]["natural"],
            mixture25=pairs["natural"]["methods"]["mixture25"],
            balanced=pairs["balanced"]["methods"]["balanced"],
        ),
        mixture25_minus_natural=pairs["natural"]["paired_delta"],
        mixture25_minus_balanced=pairs["balanced"]["paired_delta"],
        interpretation="Descriptive matched E seeds on fixed validation songs and fixed F; no significance claim.",
    )


def compare(runs, output):
    rows = []
    provenance = []
    for seed in (20261119, 20261120, 20261121):
        for method, config in (
            ("natural", "p2-natural.json"),
            ("mixture25", "p2-mixture25.json"),
            ("balanced", "p2-oracle.json"),
        ):
            complement = None
            if seed == 20261119:
                path = {
                    "natural": Path("reports/p2/kaggle-natural"),
                    "mixture25": Path("reports/p2/kaggle-mixture25"),
                    "balanced": Path("reports/p2/kaggle-oracle/full"),
                }[method]
                if method == "balanced":
                    complement = Path("reports/p2/kaggle-oracle/complementarity.csv")
            elif method == "mixture25":
                path = runs / f"mixture25-{seed}"
            else:
                path = Path("reports/p2/seed-repeat") / f"{method}-{seed}"
            record = json.loads((path / "run.json").read_text())
            expected = json.loads((Path("configs/p2") / config).read_text())
            expected["seed"] = seed
            if record["config"] != expected:
                raise ValueError(f"Configuration mismatch at {path}")
            rows.append(read_result(method, seed, path, complement))
            provenance.append(
                dict(method=method, seed=seed, path=str(path), revision=record["git_revision"])
            )
    summary = summarize(rows)
    summary["provenance"] = provenance
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/p2-mixture25-repeat"))
    parser.add_argument("--output", type=Path, default=Path("runs/p2-mixture25-repeat/analysis"))
    args = parser.parse_args()
    compare(args.runs, args.output)
