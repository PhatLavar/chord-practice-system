"""Freeze F's existing interval predictions for the oracle E comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from corpus import table
from data_e import load_events,partitions
from diagnose_f import probabilities,decode_components
from train_e import FIELDS


def prepare(run,cache,output):
    meta=partitions(cache);f=json.loads((run/"run.json").read_text())
    ids=sorted(i for i,p in meta["recordings"].items() if p=="validation")
    if f["evaluation_ids"]!=ids or f["data"]!=meta or f["config"].get("encoder","tcn")!="tcn":
        raise ValueError("F is not the expected TCN on the frozen validation data")
    _,events,empty=load_events(cache,Path("data/raw/full.tar.xz"),ids)
    rows=[];by_song={}
    for event in events:by_song.setdefault(event[0],[]).append(event)
    for rid,intervals in sorted(by_song.items()):
        with np.load(run/"predictions"/f"{rid}.npz") as data:
            p=[probabilities(data[f"logits_{j}"]) for j in range(4)]
            for _,start,end,lo,hi,label in intervals:
                pooled=[v[:,lo:hi].mean(axis=1,keepdims=True) for v in p]
                assert np.all(data["reference"][lo:hi]==label)
                rows.append([rid,start,end,label,int(decode_components(pooled)[0]),
                             *[int(v[:,0].argmax()) for v in pooled]])
    output.mkdir(parents=True,exist_ok=False)
    table(output/"events.csv"," ".join(FIELDS),rows)
    (output/"provenance.json").write_text(json.dumps(dict(F_run=str(run),F_config=f["config"],
        F_revision=f["git_revision"],events=len(rows),events_without_frames=empty,
        F_run_record_sha256=hashlib.sha256((run/"run.json").read_bytes()).hexdigest(),
        manifest_sha256=meta["manifest_sha256"],
        reference_event_accuracy=sum(r[3]==r[4] for r in rows)/len(rows),
        protocol="Mean probabilities from F on the same reference intervals supplied to E."),indent=2))
    print(f"Frozen {len(rows)} F reference events")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=Path("runs/p1-balanced-kaggle"))
    p.add_argument("--cache",type=Path,default=Path("data/processed/p1"))
    p.add_argument("--output",type=Path,default=Path("reports/p2/f-reference"))
    a=p.parse_args();prepare(a.run,a.cache,a.output)
