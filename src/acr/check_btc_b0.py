"""CPU compatibility and fixed training-window fitting check for author BTC."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from btc_adapter import BTCFrame,btc_loss,project_to_btc,predict_frames
from prepare_guitarset_cqt import load_cache,digest
from train_f import Windows


def check(cache,output):
    if output.exists():raise FileExistsError('Use a fresh B0 diagnostic directory')
    cfg=json.loads(Path('configs/baselines/btc-b0.json').read_text(encoding='utf-8'))
    vendor=Path('src/acr/vendor/btc');manifest=json.loads((vendor/'UPSTREAM.json').read_text(encoding='utf-8'))
    for name,record in manifest['files'].items():
        content=(vendor/name).read_bytes().replace(b'\r\n',b'\n')
        if hashlib.sha256(content).hexdigest()!=record['vendored_sha256_lf']:raise ValueError('Vendored source changed: '+name)
    songs,refs,meta=load_cache(cache,json.loads(Path(cfg['data_config']).read_text()),False)
    training={rid:s for rid,s in songs.items() if meta['recordings'][rid]=='train'}
    dataset=Windows(training,108,0);chosen=[];seen=set()
    for i,(rid,start) in enumerate(dataset.windows):
        y=training[rid]['y'][start:start+108];valid=y[y>0]
        if len(valid)<16:continue
        quality=int(np.bincount((valid-1)%14,minlength=14).argmax())
        if quality in seen:continue
        seen.add(quality);chosen.append(i)
        if len(chosen)==cfg['diagnostic']['windows']:break
    if len(chosen)!=cfg['diagnostic']['windows']:raise ValueError('Not enough diverse training windows')
    items=[dataset[i] for i in chosen];x=torch.stack([a for a,_ in items]);y=torch.stack([b for _,b in items]);valid=y>=0;target=project_to_btc(y)
    torch.set_num_threads(4);torch.manual_seed(cfg['seed'])
    model=BTCFrame(cfg['model']);optimizer=torch.optim.Adam(model.parameters(),lr=cfg['diagnostic']['learning_rate'])
    before=time.perf_counter();history=[];output.mkdir(parents=True)
    for step in range(1,cfg['diagnostic']['max_updates']+1):
        model.train();logits=model(x);loss=btc_loss(logits,y)
        if not torch.isfinite(loss):raise ValueError('Nonfinite BTC loss')
        optimizer.zero_grad();loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite gradient')
        optimizer.step()
        if step==1 or step%10==0:
            model.eval()
            with torch.no_grad():correct=int((model(x).argmax(1)[valid]==target[valid]).sum())
            accuracy=correct/int(valid.sum())
            row=dict(update=step,training_loss=float(loss.detach()),correct_frames=correct,clean_frame_accuracy=accuracy)
            history.append(row);print(json.dumps(row),flush=True)
            if accuracy>=cfg['diagnostic']['pass_frame_accuracy']:break
    # Test full-track tiling/last-window trimming on one selected training recording.
    rid=dataset.windows[chosen[0]][0];timeline=predict_frames(model,training[rid]['x'])
    assert timeline.shape==(170,len(training[rid]['times'])) and torch.isfinite(timeline).all()
    report=dict(stage='B0',protocol='Fresh author BTC fixed-window train-only fit; no validation selection, no pretrained weights, no B1 benchmark.',
        config=cfg,upstream_commit=manifest['commit'],vendor_manifest_sha256=digest(vendor/'UPSTREAM.json'),
        cache_metadata_sha256=digest(cache/'metadata.json'),torch=torch.__version__,numpy=np.__version__,
        parameters=sum(p.numel() for p in model.parameters()),elapsed_seconds=time.perf_counter()-before,
        windows=[dict(recording=dataset.windows[i][0],start_frame=dataset.windows[i][1]) for i in chosen],
        scored_training_frames=int(valid.sum()),history=history,
        pass_check=history[-1]['clean_frame_accuracy']>=cfg['diagnostic']['pass_frame_accuracy'],
        full_track_output_shape=list(timeline.shape),evaluation_partition='training',
        source_sha256={p.as_posix():digest(p) for p in [Path(__file__),Path('src/acr/btc_adapter.py')]})
    torch.save(dict(state_dict=model.state_dict(),config=cfg),output/'diagnostic.pt')
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=Path('data/processed/guitarset-cqt-player'))
    p.add_argument('--output',type=Path,default=Path('runs/b0-btc-compatibility'))
    a=p.parse_args();check(a.cache,a.output)
