"""Repeat natural/balanced E with matched seeds and the same frozen F."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from acr.experiments.billboard.repeat_frame import aggregate

METRICS = ("event_accuracy","macro_quality_event_recall","rare_macro_event_recall")


def read_result(method, seed, directory, complement=None):
    metrics=json.loads((directory/"metrics.json").read_text())
    reference=Path("reports/p2/f-reference/events.csv")
    if metrics["F_reference_sha256"]!=hashlib.sha256(reference.read_bytes()).hexdigest():
        raise ValueError("Run uses another F reference")
    with (complement or directory/"complementarity.csv").open() as stream:
        rows=[r for r in csv.DictReader(stream) if r["metric"]=="complete"]
    with Path("reports/p0/qualities.csv").open() as stream:
        rare={r["quality"] for r in csv.DictReader(stream) if r["rare"]=="True"}
    rare_count=sum(sum(int(r[k]) for k in ("both_correct","F_only","E_only","both_wrong"))
                   for r in rows if r["quality"] in rare)
    total=sum(sum(int(r[k]) for k in ("both_correct","F_only","E_only","both_wrong")) for r in rows)
    correct=sum(int(r["both_correct"])+int(r["E_only"]) for r in rows)
    if total!=metrics["scored_events"] or abs(correct/total-metrics["event_accuracy"])>1e-12:
        raise ValueError("Complementarity does not reproduce event accuracy")
    return dict(method=method,seed=seed,**{k:metrics[k] for k in METRICS},
                correct_events=correct,scored_events=total,rare_reference_events=rare_count,
                rare_E_only_correct=sum(int(r["E_only"]) for r in rows if r["quality"] in rare),
                rare_correct_events=sum(int(r["both_correct"])+int(r["E_only"]) for r in rows if r["quality"] in rare))


def run(plan_path,output,device):
    plan=json.loads(plan_path.read_text());seeds=plan["seeds"]
    if not seeds or any(type(s) is not int or s<0 for s in seeds) or len(set(seeds))!=len(seeds) or 20261119 in seeds:
        raise ValueError("Use distinct additional seeds")
    configs={method:json.loads(Path(plan[f"{method}_config"]).read_text()) for method in ("natural","balanced")}
    if {k:v for k,v in configs["natural"].items() if k!="sampling"}!={k:v for k,v in configs["balanced"].items() if k!="sampling"}:
        raise ValueError("Configs differ beyond sampling")
    if configs["natural"]["sampling"]!="natural" or configs["balanced"]["sampling"]!="quality_song_mixture":
        raise ValueError("Unexpected sampling pair")
    revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    record=dict(plan=plan,git_revision=revision,device=device,original_seed=20261119,
                F_reference_sha256=hashlib.sha256(Path("reports/p2/f-reference/events.csv").read_bytes()).hexdigest())
    if (output/"plan.json").exists() and json.loads((output/"plan.json").read_text())!=record:
        raise ValueError("Output root belongs to another plan/revision/device/reference")
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for method,directory,complement in (
        ("natural",Path("reports/p2/kaggle-natural"),None),
        ("balanced",Path("reports/p2/kaggle-oracle/full"),Path("reports/p2/kaggle-oracle/complementarity.csv"))):
        original=json.loads((directory/"run.json").read_text())
        if original["config"]!=configs[method]:
            raise ValueError("Original E configuration does not match the planned comparison")
        rows.append(read_result(method,20261119,directory,complement))
    (output/"plan.json").write_text(json.dumps(record,indent=2))
    for seed in seeds:
        for method in ("natural","balanced"):
            cfg=dict(configs[method],seed=seed)
            config=output/f"{method}-{seed}.json"
            config.write_text(json.dumps(cfg,indent=2))
            target=output/f"{method}-{seed}"
            if target.exists():
                if not (target/"metrics.json").is_file() or not (target/"complementarity.csv").is_file():
                    raise ValueError(f"Incomplete E run: {target}. Preserve it and use a fresh output root or remove only the failed subdirectory before retrying.")
                previous=json.loads((target/"run.json").read_text())
                if previous["config"]!=cfg or previous["git_revision"]!=revision:
                    raise ValueError(f"Existing run has a different config/revision: {target}")
                print(f"Reusing completed E run {target}",flush=True)
            else:
                subprocess.run([sys.executable,'-m', 'acr.training.event',"--config",str(config),
                                "--output",str(target),"--device",device],check=True)
            rows.append(read_result(method,seed,target))
            (output/"completed.json").write_text(json.dumps(rows,indent=2))
    with (output/"per_seed.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    summary=aggregate(rows,metrics=METRICS,methods=("natural","balanced"))
    summary.update(delta_direction="balanced minus natural",
                   original_pair_source="Verified P2 natural and balanced seed 20261119",
                   new_pair_revision=revision,F_reference_sha256=record["F_reference_sha256"],
                   interpretation="Three E seeds on fixed validation songs; F stays fixed at its original balanced checkpoint. This does not measure variation across F/E seed pairs.")
    (output/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,default=Path("configs/p2/p2-seeds.json"))
    parser.add_argument("--output",type=Path,default=Path("runs/p2-seed-repeat"))
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    args=parser.parse_args()
    run(args.plan,args.output,args.device)
