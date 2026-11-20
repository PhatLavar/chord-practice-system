"""Load gold intervals on P1's normalized cache, with event-level sampling."""
import csv
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from diagnose_f import reference_events
from p1_data import CLASSES, QUALITIES


def load_events(cache, labels, ids):
    meta=json.loads((cache/"metadata.json").read_text())
    with labels.open("rb") as f:
        if hashlib.file_digest(f,"sha256").hexdigest()!=meta["source_sha256"]["full.tar.xz"]:
            raise ValueError("Detailed label checksum mismatch")
    songs={};events=[];empty=0
    with tarfile.open(labels) as archive:
        members={Path(m.name).parent.name:m for m in archive if m.name.endswith("/full.lab")}
        for rid in ids:
            with np.load(cache/f"{rid}.npz") as data:
                songs[rid]={k:data[k] for k in ("x","y","times")}
            times=songs[rid]["times"]
            for start,end,label in reference_events(archive.extractfile(members[rid]).read().decode().splitlines()):
                if label<0:
                    continue
                lo,hi=np.searchsorted(times,[start,end],side="left")
                if lo==hi:
                    empty+=1
                    continue
                if not np.all(songs[rid]["y"][lo:hi]==label):
                    raise ValueError(f"Cache/reference mismatch for {rid}")
                events.append((rid,start,end,int(lo),int(hi),label))
    return songs,events,empty


def partitions(cache):
    meta=json.loads((cache/"metadata.json").read_text())
    if meta["classes"]!=CLASSES or meta["manifest_sha256"]!=hashlib.sha256(Path("splits/billboard-v1.csv").read_bytes()).hexdigest():
        raise ValueError("Cache does not match the frozen vocabulary/split")
    with Path("splits/billboard-v1.csv").open() as f:
        rows=list(csv.DictReader(f))
    for part in ("train_labeled","validation"):
        expected={r["id"] for r in rows if r["partition"]==part and r["id"]==r["representative"]}
        actual={i for i,p in meta["recordings"].items() if p==part}
        if actual!=expected:
            raise ValueError(f"Cache recording mismatch for {part}")
    return meta


class Events(Dataset):
    def __init__(self,songs,events,balanced=False,context=31,balanced_probability=.5):
        if not 0 <= balanced_probability <= 1:
            raise ValueError("Balanced probability must be between zero and one")
        self.songs,self.events,self.balanced,self.context=songs,events,balanced,context
        self.balanced_probability=balanced_probability
        self.groups={}
        for index,event in enumerate(events):
            label=event[-1]
            if label>0:
                self.groups.setdefault((label-1)%len(QUALITIES),{}).setdefault(event[0],[]).append(index)
        self.qualities=sorted(self.groups)
        if balanced and not self.qualities:
            raise ValueError("No supported chord events for balanced sampling")

    def __len__(self):
        return len(self.events)

    def __getitem__(self,index):
        if self.balanced and torch.rand(()).item()>=1-self.balanced_probability:
            quality=self.qualities[torch.randint(len(self.qualities),()).item()]
            groups=list(self.groups[quality].values())
            candidates=groups[torch.randint(len(groups),()).item()]
            index=candidates[torch.randint(len(candidates),()).item()]
        rid,_,_,lo,hi,label=self.events[index]
        source=self.songs[rid]["x"]
        x=np.zeros((source.shape[1],hi-lo+2*self.context),dtype=np.float32)
        left,right=max(0,lo-self.context),min(len(source),hi+self.context)
        offset=left-lo+self.context
        x[:,offset:offset+right-left]=source[left:right].T
        mask=np.zeros(x.shape[-1],dtype=bool)
        mask[self.context:self.context+hi-lo]=True
        return torch.from_numpy(x),torch.from_numpy(mask),label


def collate_events(items):
    length=max(x.shape[-1] for x,_,_ in items)
    channels=items[0][0].shape[0]
    if any(features.shape[0]!=channels for features,_,_ in items):
        raise ValueError("A batch must contain one input representation")
    x=torch.zeros(len(items),channels,length)
    mask=torch.zeros(len(items),length,dtype=torch.bool)
    for i,(features,pooling,_) in enumerate(items):
        x[i,:,:features.shape[-1]]=features
        mask[i,:len(pooling)]=pooling
    return x,mask,torch.tensor([label for _,_,label in items]).unsqueeze(1)
