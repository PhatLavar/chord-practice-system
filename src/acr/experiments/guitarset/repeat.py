"""Matched-seed natural/transposition F/E repeats on the frozen player cache."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

from acr.data.guitarset.cqt import digest, load_cache

SEEDS = (20261120, 20261121)


def aggregate(rows):
    result = {}
    for teacher in ("F", "E"):
        pairs = [r for r in rows if r["teacher"] == teacher]
        result[teacher] = {}
        for metric in ("root", "majmin", "triads", "tetrads", "event_accuracy"):
            entry = {}
            for arm in ("natural", "transpose"):
                values = [r[arm][metric] for r in pairs]
                entry[arm] = dict(
                    mean=statistics.mean(values),
                    sample_std=statistics.stdev(values) if len(values) > 1 else None,
                )
            deltas = [r["transpose"][metric] - r["natural"][metric] for r in pairs]
            entry["paired_delta"] = dict(
                mean=statistics.mean(deltas),
                sample_std=statistics.stdev(deltas) if len(deltas) > 1 else None,
                values=deltas,
            )
            result[teacher][metric] = entry
    return result


def scores(metrics):
    return {
        **{k: v["score"] for k, v in metrics["metrics"].items()},
        "event_accuracy": metrics["supported_oracle_event_accuracy"],
    }


def run(cache, output, device, smoke=False):
    output.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path("configs/p2/p2-guitarset-cqt-player.json").read_text())
    songs, refs, meta = load_cache(cache, base, smoke)
    del songs, refs
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    sources = {p.as_posix(): digest(p) for p in Path("src/acr").rglob("*.py")}
    configs = output / "configs"
    configs.mkdir(exist_ok=True)
    historical = {}
    historical_hashes = {}
    for teacher in ("F", "E"):
        historical[teacher] = {}
        for arm, folder in [("natural", "player-results"), ("transpose", "transposition/results")]:
            path = Path("reports/p2/guitarset-cqt") / folder / (teacher + "_metrics.json")
            historical_hashes[path.as_posix()] = digest(path)
            historical[teacher][arm] = json.loads(path.read_text())
    plan = dict(
        seeds=list(SEEDS),
        smoke=smoke,
        git_revision=revision,
        source_sha256=sources,
        cache_metadata_sha256=digest(cache / "metadata.json"),
        historical_reference_seed=20261119,
        historical_metrics_sha256=historical_hashes,
    )
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing repeat run has different provenance")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    rows = []
    if not smoke:
        rows = [
            dict(seed=20261119, teacher=t, **{arm: scores(m) for arm, m in historical[t].items()})
            for t in ("F", "E")
        ]
    for seed in SEEDS:
        for arm, filename in [
            ("natural", "p2-guitarset-cqt-player.json"),
            ("transpose", "p2-guitarset-cqt-transpose.json"),
        ]:
            cfg = json.loads((Path("configs/p2") / filename).read_text())
            cfg.update(seed=seed, cache_config="configs/p2/p2-guitarset-cqt-player.json")
            config = configs / f"{arm}-{seed}.json"
            if config.exists() and json.loads(config.read_text()) != cfg:
                raise ValueError("Config changed")
            config.write_text(json.dumps(cfg, indent=2) + "\n")
            directory = output / f"{arm}-{seed}"
            if directory.exists():
                if not (directory / "summary.json").exists():
                    raise FileExistsError(f"Incomplete run; preserve and inspect: {directory}")
                for teacher in ("F", "E"):
                    record = json.loads((directory / teacher / "run.json").read_text())
                    if (
                        record["config"] != cfg
                        or record["smoke"] != smoke
                        or record["git_revision"] != revision
                        or record["source_sha256"] != sources
                        or record["cache_metadata"] != meta
                    ):
                        raise ValueError("Completed stage provenance changed")
            else:
                command = [
                    sys.executable,
                    "-m",
                    "acr.training.guitarset",
                    "--cache",
                    str(cache),
                    "--config",
                    str(config),
                    "--output",
                    str(directory),
                    "--device",
                    device,
                ]
                if smoke:
                    command.append("--smoke")
                subprocess.run(command, check=True)
        for teacher in ("F", "E"):
            row = dict(seed=seed, teacher=teacher)
            for arm in ("natural", "transpose"):
                row[arm] = scores(
                    json.loads((output / f"{arm}-{seed}" / teacher / "metrics.json").read_text())
                )
            rows.append(row)
    report = dict(
        seeds=list(SEEDS) if smoke else [20261119, *SEEDS],
        smoke=smoke,
        protocol="Matched-seed development repeats. Historical first seed retained only in full aggregate; never choose best seed.",
        rows=rows,
        aggregate=aggregate(rows),
    )
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    run(a.cache, a.output, a.device, a.smoke)
