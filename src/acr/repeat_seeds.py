"""Run paired baseline/balanced seeds and summarize with the original pair."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


METRICS = ("frame_accuracy", "macro_chord_recall", "event_accuracy", "rare_macro_event_recall")


def result(name, seed, metrics, diagnostic):
    return dict(method=name, seed=seed,
                **{key: metrics[key] if key in metrics else diagnostic[key] for key in METRICS})


def aggregate(rows):
    pairs = {}
    for row in rows:
        pair = pairs.setdefault(row["seed"], {})
        if row["method"] in pair:
            raise ValueError("Duplicate method/seed result")
        pair[row["method"]] = row
    if not pairs or any(set(pair) != {"baseline", "balanced"} for pair in pairs.values()):
        raise ValueError("Every seed needs both methods")
    summary = dict(seeds=sorted(pairs), n=len(pairs), methods={}, paired_delta={})
    for method in ("baseline", "balanced"):
        summary["methods"][method] = {}
        for metric in METRICS:
            values = np.array([pairs[seed][method][metric] for seed in sorted(pairs)])
            if not np.isfinite(values).all() or np.any((values<0)|(values>1)):
                raise ValueError("Metrics must be finite proportions")
            summary["methods"][method][metric] = dict(mean=float(values.mean()),
                sample_std=float(values.std(ddof=1)) if len(values)>1 else None)
    for metric in METRICS:
        values = np.array([pair["balanced"][metric]-pair["baseline"][metric] for pair in pairs.values()])
        summary["paired_delta"][metric] = dict(mean=float(values.mean()),
            sample_std=float(values.std(ddof=1)) if len(values)>1 else None,
            positive_seeds=int((values>0).sum()), negative_seeds=int((values<0).sum()),
            tied_seeds=int((values==0).sum()))
    return summary


def run(plan_path, output, device):
    plan = json.loads(plan_path.read_text())
    seeds = plan["seeds"]
    if not seeds or any(type(seed) is not int or seed<0 for seed in seeds) or len(set(seeds)) != len(seeds) or 20261119 in seeds:
        raise ValueError("Use distinct additional seeds, excluding the original seed")
    configs = {method:json.loads(Path(plan[f"{method}_config"]).read_text())
               for method in ("baseline", "balanced")}
    if {k:v for k,v in configs["balanced"].items() if k!="sampling"} != configs["baseline"]:
        raise ValueError("Configs differ beyond sampling")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    # Reuse the already verified original-seed results, not another training run.
    for method, base, diag in (
        ("baseline", "reports/p1/kaggle-baseline", "reports/p1/kaggle-diagnostics"),
        ("balanced", "reports/p1/kaggle-balanced", "reports/p1/balanced-diagnostics")):
        original = json.loads((Path(base)/"run.json").read_text())
        assert original["config"]["seed"] == 20261119
        assert original["config"] == configs[method]
        rows.append(result(method, 20261119,
            json.loads((Path(base)/"metrics.json").read_text()),
            json.loads((Path(diag)/"summary.json").read_text())))
    plan_record = dict(plan=plan,git_revision=revision,original_seed=20261119,device=device)
    if (output/"plan.json").exists():
        if json.loads((output/"plan.json").read_text()) != plan_record:
            raise ValueError("Output root belongs to another plan/revision/device")
    (output/"plan.json").write_text(json.dumps(plan_record,indent=2))
    for seed in seeds:
        for method in ("baseline", "balanced"):
            cfg = dict(configs[method], seed=seed)
            config_path = output/f"{method}-{seed}.json"
            config_path.write_text(json.dumps(cfg,indent=2))
            target = output/f"{method}-{seed}"
            if target.exists():
                if not (target/"metrics.json").is_file():
                    raise ValueError(f"Incomplete run {target}; retain it and choose a new output root, or remove only that failed run before retrying")
                recorded = json.loads((target/"run.json").read_text())
                if recorded["config"] != cfg or recorded["git_revision"] != revision:
                    raise ValueError(f"Existing run has a different config/revision: {target}")
                print(f"Reusing completed run {target}",flush=True)
            else:
                subprocess.run([sys.executable,"src/acr/train_f.py","--config",str(config_path),
                                "--output",str(target),"--device",device],check=True)
            diagnostic = target/"diagnostics"
            if not (diagnostic/"summary.json").is_file():
                if diagnostic.exists():
                    raise ValueError(f"Incomplete diagnostic: {diagnostic}")
                subprocess.run([sys.executable,"src/acr/diagnose_f.py","--run",str(target),
                                "--output",str(diagnostic)],check=True)
            rows.append(result(method,seed,json.loads((target/"metrics.json").read_text()),
                               json.loads((diagnostic/"summary.json").read_text())))
            # Partial progress remains readable if a later run is interrupted.
            (output/"completed.json").write_text(json.dumps(rows,indent=2))
    with (output/"per_seed.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["method","seed",*METRICS])
        writer.writeheader();writer.writerows(rows)
    summary=aggregate(rows)
    summary["original_pair_source"]="Verified Kaggle outputs at 8ec616c (baseline) and d250916 (balanced)"
    summary["new_pair_revision"]=revision
    summary["interpretation"]="Descriptive three-seed comparison on the same validation songs; no significance claim."
    (output/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,default=Path("configs/p1-seeds.json"))
    parser.add_argument("--output",type=Path,default=Path("runs/p1-seed-repeat"))
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    args=parser.parse_args()
    run(args.plan,args.output,args.device)
