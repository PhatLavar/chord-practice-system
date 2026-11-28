"""Audit train-only quality exposure before and after one sampled epoch."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from acr.data.billboard.corpus import table
from acr.vocabulary import QUALITIES, CLASSES
from acr.data.windows import Windows, QualityBalancedWindows


def audit(cache, output):
    meta=json.loads((cache/"metadata.json").read_text())
    if meta["classes"] != CLASSES:
        raise ValueError("Cache vocabulary mismatch")
    songs={}
    for rid,part in sorted(meta["recordings"].items()):
        if part=="train_labeled":
            with np.load(cache/f"{rid}.npz") as d:
                songs[rid]={"x":d["x"],"y":d["y"]}
    natural=Windows(songs,512,31)
    balanced=QualityBalancedWindows(songs,512,31)
    counts=np.zeros((14,4),dtype=np.int64)
    for song in songs.values():
        y=song["y"]
        for q in range(14):
            mask=(y>0)&((y-1)%14==q)
            counts[q,0]+=mask.sum()
            counts[q,1]+=mask.any()
    for rid,start in natural.windows:
        y=songs[rid]["y"][start:start+512]
        for q in range(14):
            counts[q,2]+=np.any((y>0)&((y-1)%14==q))
    sampled=np.zeros(14,dtype=np.int64)
    sampled_windows=np.zeros(14,dtype=np.int64)
    torch.manual_seed(20261119)
    for index in range(len(balanced)):
        _,y=balanced[index]
        for q in range(14):
            n=int(((y>0)&((y-1)%14==q)).sum())
            sampled[q]+=n;sampled_windows[q]+=n>0
    output.mkdir(parents=True,exist_ok=False)
    table(output/"quality_exposure.csv",
          "quality source_frames source_songs natural_windows sampled_frames sampled_windows frame_exposure_ratio",
          [[q,*map(int,counts[j,:3]),int(sampled[j]),int(sampled_windows[j]),
            float(sampled[j]/counts[j,0]) if counts[j,0] else ""]
           for j,q in enumerate(QUALITIES)])
    summary=dict(labeled_train_songs=len(songs),windows_per_epoch=len(natural),
                 seed=20261119,manifest_sha256=meta["manifest_sha256"],
                 diagnostic="One simulated epoch; sampling with replacement; not new independent data",
                 policy="50% natural windows; 50% uniform observed quality, uniform song, uniform matching frame",
                 source_chord_frames=int(counts[:,0].sum()),sampled_chord_frames=int(sampled.sum()))
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,default=Path("data/processed/p1"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    audit(args.cache,args.output)
