"""Train frame teacher F on frozen Billboard representatives."""
import argparse
import csv
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from p1_data import CLASSES, QUALITIES
from teacher_f import FrameTeacher, decode, losses, targets
from p1_data import TRIADS, EXTENSIONS
from transformer_f import TransformerFrameTeacher


def component_weights(train_songs, device):
    counts = [torch.zeros(size, dtype=torch.int64) for size in (len(TRIADS), len(EXTENSIONS))]
    for song in train_songs.values():
        for count, target in zip(counts, targets(torch.from_numpy(song["y"]))[2:]):
            observed = target[target >= 0]
            count += torch.bincount(observed, minlength=len(count))
    weights = []
    for count in counts:
        observed = count > 0
        if not observed.any():
            raise ValueError("No observed component targets in labeled training songs")
        # Inverse square root reduces majority-class dominance with bounded ratios.
        weight = torch.zeros(len(count), dtype=torch.float32)
        weight[observed] = (count[observed].max() / count[observed].float()).sqrt().clamp(max=10)
        weight /= (weight * count).sum() / count.sum()
        weights.append(weight.to(device))
    record = dict(method="inverse_sqrt",max_ratio=10,
                  triad=dict(labels=TRIADS,counts=counts[0].tolist(),weights=weights[0].cpu().tolist()),
                  extension=dict(labels=EXTENSIONS,counts=counts[1].tolist(),weights=weights[1].cpu().tolist()))
    return [None, None, *weights], record


class Windows(Dataset):
    def __init__(self, songs, length, context, include_padding_mask=False):
        self.songs, self.length, self.context = songs, length, context
        self.include_padding_mask = include_padding_mask
        self.windows = [(rid, start) for rid, song in songs.items()
                        for start in range(0, len(song["y"]), length)
                        if np.any(song["y"][start:start+length] >= 0)]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        rid, start = self.windows[index]
        return self.get_window(rid, start)

    def get_window(self, rid, start):
        song = self.songs[rid]
        stop = min(start+self.length, len(song["y"]))
        x = np.zeros((24, self.length+2*self.context), dtype=np.float32)
        lo, hi = max(0,start-self.context), min(len(song["y"]),start+self.length+self.context)
        offset = lo-start+self.context
        x[:,offset:offset+hi-lo] = song["x"][lo:hi].T
        y = np.full(self.length, -100, dtype=np.int64)
        y[:stop-start] = song["y"][start:stop]
        if self.include_padding_mask:
            padding = np.ones(x.shape[-1], dtype=bool)
            padding[offset:offset+hi-lo] = False
            return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(padding)
        return torch.from_numpy(x), torch.from_numpy(y)


class QualityBalancedWindows(Windows):
    """Half natural windows, half quality/song-balanced centered windows."""

    def __init__(self, songs, length, context, include_padding_mask=False):
        super().__init__(songs, length, context, include_padding_mask)
        self.anchors = {}
        for rid, song in songs.items():
            y = song["y"]
            for quality in range(len(QUALITIES)):
                indices = np.flatnonzero((y > 0) & ((y-1) % len(QUALITIES) == quality))
                if len(indices):
                    self.anchors.setdefault(quality, []).append((rid, indices))
        self.qualities = sorted(self.anchors)
        if not self.qualities:
            raise ValueError("Balanced sampling requires observed chord qualities")

    def __getitem__(self, index):
        if torch.rand(()).item() < .5:
            return super().__getitem__(index)
        quality = self.qualities[torch.randint(len(self.qualities), ()).item()]
        entries = self.anchors[quality]
        rid, indices = entries[torch.randint(len(entries), ()).item()]
        anchor = int(indices[torch.randint(len(indices), ()).item()])
        start = max(0, min(anchor-self.length//2, len(self.songs[rid]["y"])-self.length))
        return self.get_window(rid, start)


@torch.no_grad()
def recording_logits(model, rid, song, device):
    if isinstance(model, TransformerFrameTeacher):
        windows = Windows({rid:song},model.window,model.context,include_padding_mask=True)
        collected = [[] for _ in range(4)]
        # Visit every frame, including windows with entirely masked reference labels.
        for start in range(0,len(song["y"]),model.window):
            x, _, padding = windows.get_window(rid,start)
            heads = model(x.unsqueeze(0).to(device),padding.unsqueeze(0).to(device))
            size = min(model.window,len(song["y"])-start)
            for chunks,head in zip(collected,heads):
                chunks.append(head[:,:,model.context:model.context+size].cpu())
        return [torch.cat(chunks,dim=2) for chunks in collected]
    return model(torch.from_numpy(song["x"].T.copy()).unsqueeze(0).to(device))


@torch.no_grad()
def evaluate(model, songs, device, output=None):
    model.eval()
    confusion = np.zeros((len(CLASSES),len(CLASSES)), dtype=np.int64)
    song_rows = []
    for rid, song in songs.items():
        heads = recording_logits(model,rid,song,device)
        pred = decode(heads)[0].cpu().numpy()
        y = song["y"]
        valid = y >= 0
        np.add.at(confusion, (y[valid],pred[valid]), 1)
        song_rows.append([rid,int(valid.sum()),int((pred[valid]==y[valid]).sum())])
        if output:
            np.savez_compressed(output/f"{rid}.npz", times=song["times"], reference=y,
                                prediction=pred, **{f"logits_{i}":h[0].cpu().numpy() for i,h in enumerate(heads)})
    support = confusion.sum(1)
    correct = confusion.diagonal()
    metrics = dict(frame_accuracy=float(correct.sum()/max(1,support.sum())),
                   macro_chord_recall=float(np.mean(correct[support>0]/support[support>0])),
                   scored_frames=int(support.sum()))
    if output:
        np.save(output/"confusion.npy",confusion)
        with (output/"per_song.csv").open("w",newline="") as f:
            w=csv.writer(f);w.writerow(["id","scored_frames","correct_frames"]);w.writerows(song_rows)
        with (output/"per_quality.csv").open("w",newline="") as f:
            w=csv.writer(f);w.writerow(["quality","frames","correct_complete_chord_frames","complete_chord_recall","complete_chord_precision"])
            for j,q in enumerate(QUALITIES):
                ids=np.arange(1+j,len(CLASSES),len(QUALITIES))
                n=int(support[ids].sum());c=int(correct[ids].sum())
                predicted=int(confusion[:,ids].sum())
                w.writerow([q,n,c,c/n if n else "",c/predicted if predicted else ""])
    return metrics


def train(cache, config, output, device):
    cfg=json.loads(config.read_text())
    meta=json.loads((cache/"metadata.json").read_text())
    current_manifest=hashlib.sha256(Path("splits/billboard-v1.csv").read_bytes()).hexdigest()
    if meta["manifest_sha256"]!=current_manifest or meta["classes"]!=CLASSES:
        raise ValueError("Cache does not match the frozen split/vocabulary")
    random.seed(cfg["seed"]);np.random.seed(cfg["seed"]);torch.manual_seed(cfg["seed"])
    torch.set_num_threads(4)
    if device=="cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; enable a Kaggle GPU or use --device cpu")
    train_ids=sorted(i for i,p in meta["recordings"].items() if p=="train_labeled")
    if cfg["mode"]=="overfit":
        train_ids=train_ids[:cfg["songs"]]
        eval_ids=train_ids
    elif cfg["mode"]=="baseline":
        eval_ids=sorted(i for i,p in meta["recordings"].items() if p=="validation")
    else:
        raise ValueError("Unknown mode")
    songs={}
    for rid in set(train_ids+eval_ids):
        with np.load(cache/f"{rid}.npz") as data:
            songs[rid]={k:data[k] for k in ("x","y","times")}
    train_songs={i:songs[i] for i in train_ids}
    eval_songs={i:songs[i] for i in eval_ids}
    weighting = cfg.get("component_weighting", "none")
    if weighting not in ("none", "inverse_sqrt"):
        raise ValueError("Unknown component weighting")
    class_weights, weight_record = None, None
    if weighting == "inverse_sqrt":
        class_weights, weight_record = component_weights(train_songs, device)
    encoder = cfg.get("encoder","tcn")
    if encoder not in ("tcn","transformer"):
        raise ValueError("Unknown encoder")
    model = (TransformerFrameTeacher(cfg["width"],cfg["window"]) if encoder=="transformer"
             else FrameTeacher(cfg["width"])).to(device)
    sampling = cfg.get("sampling", "natural")
    if sampling not in ("natural", "quality_song_mixture"):
        raise ValueError("Unknown sampling policy")
    dataset_class = QualityBalancedWindows if sampling == "quality_song_mixture" else Windows
    dataset = dataset_class(train_songs,cfg["window"],model.context,
                            include_padding_mask=encoder=="transformer")
    loader=DataLoader(dataset,
                      batch_size=cfg["batch_size"],shuffle=True,num_workers=0)
    output.mkdir(parents=True,exist_ok=False)
    revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    dirty=bool(subprocess.check_output(["git","status","--porcelain"],text=True).strip())
    record=dict(config=cfg,train_ids=train_ids,evaluation_ids=eval_ids,
                evaluation_partition="training" if cfg["mode"]=="overfit" else "validation",
                git_revision=revision,git_dirty=dirty,torch=torch.__version__,numpy=np.__version__,
                device=device,parameters=sum(p.numel() for p in model.parameters()),data=meta,
                source_code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(Path("src/acr").glob("*.py"))})
    (output/"run.json").write_text(json.dumps(record,indent=2))
    if sampling == "quality_song_mixture":
        sampling_record = dict(policy=sampling, natural_probability=.5,
                               balanced_probability=.5, windows_per_epoch=len(dataset),
                               optimizer_steps_per_epoch=len(loader),
                               supported_qualities=[QUALITIES[q] for q in dataset.qualities],
                               train_song_support={QUALITIES[q]:len(entries) for q,entries in dataset.anchors.items()})
        (output/"sampling.json").write_text(json.dumps(sampling_record,indent=2))
    if weight_record is not None:
        (output/"loss_weights.json").write_text(json.dumps(weight_record,indent=2))
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg["learning_rate"])
    best=-1; history=[]
    for epoch in range(1,cfg["epochs"]+1):
        started=time.perf_counter();model.train()
        sums=np.zeros(4);batches=0
        quality_exposure=np.zeros(len(QUALITIES),dtype=np.int64)
        for batch in loader:
            x,y = batch[:2]
            if sampling == "quality_song_mixture":
                observed=y[y>0]
                quality_exposure += torch.bincount((observed-1)%len(QUALITIES),
                                                  minlength=len(QUALITIES)).numpy()
            x,y=x.to(device),y.to(device)
            logits = model(x,batch[2].to(device)) if encoder=="transformer" else model(x)
            heads=[h[:,:,model.context:model.context+cfg["window"]] for h in logits]
            terms=losses(heads,y,class_weights);loss=sum(terms)
            if not torch.isfinite(loss):raise ValueError("Nonfinite loss")
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step();sums += [t.item() for t in terms];batches+=1
        metrics=evaluate(model,eval_songs,device)
        row=dict(epoch=epoch,head_losses=(sums/max(1,batches)).tolist(),
                 seconds=time.perf_counter()-started,**metrics)
        if sampling == "quality_song_mixture":
            row["sampled_quality_frames"] = dict(zip(QUALITIES,map(int,quality_exposure)))
        history.append(row);print(json.dumps(row),flush=True)
        (output/"history.json").write_text(json.dumps(history,indent=2))
        if metrics["frame_accuracy"]>best:
            best=metrics["frame_accuracy"]
            torch.save(dict(state_dict=model.state_dict(),width=cfg["width"],classes=CLASSES,epoch=epoch,
                            encoder=encoder,window=cfg["window"]),output/"best.pt")
    checkpoint=torch.load(output/"best.pt",map_location=device,weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    predictions=output/"predictions";predictions.mkdir()
    final=evaluate(model,eval_songs,device,predictions)
    final.update(best_epoch=checkpoint["epoch"],evaluation_partition=record["evaluation_partition"],
                 overfit_pass=best>=.95 if cfg["mode"]=="overfit" else None,
                 peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=="cuda" else None)
    (output/"metrics.json").write_text(json.dumps(final,indent=2))
    print(json.dumps(final),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,default=Path("data/processed/p1"))
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    a=p.parse_args();train(a.cache,a.config,a.output,a.device)
