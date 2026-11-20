"""Matched frame-teacher control and JAAH auxiliary-input experiment."""
import argparse
import csv
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from corpus import table
from data_e import load_events, partitions
from diagnose_f import probabilities, decode_components
from experiment_jaah import MultiInputFrameTeacher, load_aux, replacement_steps
from p1_data import CLASSES, QUALITIES
from prepare_jaah import digest
from teacher_f import losses
from train_e import FIELDS
from train_f import QualityBalancedWindows, evaluate


def pool_events(predictions, events, output):
    """Mean frame probabilities, never softmax of mean logits."""
    pooled = [[] for _ in range(4)]; current = None
    for rid, start, end, lo, hi, label in events:
        if rid != current:
            with np.load(predictions/f'{rid}.npz') as data:
                reference = data['reference']
                probs = [probabilities(data[f'logits_{j}']) for j in range(4)]
                times = data['times']
            current = rid
        if (lo, hi) != tuple(np.searchsorted(times, [start,end], side='left')) or not np.all(reference[lo:hi] == label):
            raise ValueError('Saved frame data differs from reference intervals')
        for j in range(4): pooled[j].append(probs[j][:,lo:hi].mean(1))
    pooled = [np.stack(v,axis=1) for v in pooled]
    y = np.array([e[-1] for e in events]); pred = decode_components(pooled)
    rows = [[e[0],e[1],e[2],e[-1],int(pred[i]),*[int(p[:,i].argmax()) for p in pooled]] for i,e in enumerate(events)]
    table(output/'events.csv',' '.join(FIELDS),rows)
    np.savez_compressed(output/'event_probabilities.npz',reference=y,prediction=pred,
        **{f'probabilities_{j}':p for j,p in enumerate(pooled)})


def run(config, cache, aux_cache, output, device, smoke=False):
    if output.exists(): raise FileExistsError('Use a fresh F output directory')
    cfg = json.loads(config.read_text())
    if cfg['arms'] != ['billboard_control','jaah_mixed'] or cfg['seed'] != 20261119:
        raise ValueError('Expected the prespecified primary seed and matched arms')
    if device == 'cuda' and not torch.cuda.is_available(): raise ValueError('Enable a GPU')
    torch.set_num_threads(4); meta = partitions(cache)
    train_ids = sorted(i for i,p in meta['recordings'].items() if p=='train_labeled')
    eval_ids = sorted(i for i,p in meta['recordings'].items() if p=='validation')
    if smoke: train_ids,eval_ids = train_ids[:2],eval_ids[:2]
    songs,events,_ = load_events(cache,Path('data/raw/full.tar.xz'),sorted(train_ids+eval_ids))
    train_songs = {i:songs[i] for i in train_ids}; val_songs = {i:songs[i] for i in eval_ids}
    val_events = [e for e in events if e[0] in val_songs]
    if not smoke:
        with Path('reports/p2/f-reference/events.csv').open() as stream:
            frozen = [(r['id'],float(r['start']),float(r['end']),int(r['reference'])) for r in csv.DictReader(stream)]
        if frozen != [(e[0],e[1],e[2],e[-1]) for e in val_events]:
            raise ValueError('Validation intervals differ from frozen Billboard protocol')
    aux_songs,_,aux_meta = load_aux(aux_cache)
    output.mkdir(parents=True); outcomes = {}
    for arm in cfg['arms']:
        random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
        if device=='cuda': torch.cuda.reset_peak_memory_stats()
        model = MultiInputFrameTeacher(cfg['width']).to(device)
        training = QualityBalancedWindows(train_songs,cfg['window'],model.context)
        auxiliary = QualityBalancedWindows(aux_songs,cfg['window'],model.context)
        loader = DataLoader(training,batch_size=cfg['batch_size'],shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
        steps = min(4,len(loader)) if smoke else len(loader); epochs = 2 if smoke else cfg['epochs']
        directory = output/arm; directory.mkdir()
        record = dict(config=cfg,arm=arm,teacher='F',smoke=smoke,data=meta,auxiliary=aux_meta,
            train_ids=train_ids,evaluation_ids=eval_ids,evaluation_partition='validation_subset_smoke' if smoke else 'validation',
            steps_per_epoch=steps,epochs=epochs,windows_per_epoch=len(training),
            parameters=sum(p.numel() for p in model.parameters()),device=device,torch=torch.__version__,numpy=np.__version__,
            git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
            source_code_sha256={p.as_posix():digest(p) for p in Path('src/acr').glob('*.py')},
            protocol='50% quality/song balanced windows within each source; replace auxiliary batches; select by Billboard frame accuracy')
        (directory/'run.json').write_text(json.dumps(record,indent=2))
        best=-1.;history=[]
        for epoch in range(1,epochs+1):
            # Two-song smoke has only two batches; 25% rounds to zero. Exercise both paths.
            fraction = (.5 if smoke else cfg['auxiliary_batch_fraction']) if arm=='jaah_mixed' else 0.
            selected = replacement_steps(steps,fraction,cfg['seed'],epoch)
            model.train();sums=np.zeros(4);started=time.perf_counter()
            exposure={s:np.zeros(15,dtype=np.int64) for s in ('billboard','jaah')}
            windows={s:0 for s in exposure};masked={s:0 for s in exposure}
            for step,(x,y) in enumerate(loader):
                if step>=steps:break
                source='billboard'
                if step in selected:
                    source='jaah'
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(cfg['seed']+epoch*10000+step)
                        indices=torch.randint(len(auxiliary),(len(y),)).tolist()
                        batch=[auxiliary[i] for i in indices]
                        x=torch.stack([b[0] for b in batch]);y=torch.stack([b[1] for b in batch])
                observed=y[y>=0];buckets=torch.where(observed==0,14,(observed-1)%14)
                exposure[source]+=torch.bincount(buckets,minlength=15).numpy()
                windows[source]+=len(y);masked[source]+=int((y<0).sum())
                heads=[h[:,:,model.context:model.context+cfg['window']] for h in model(x.to(device))]
                terms=losses(heads,y.to(device));loss=sum(terms)
                if not torch.isfinite(loss):raise ValueError('Nonfinite F loss')
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                sums += [v.item() for v in terms]
            metrics=evaluate(model,val_songs,device)
            row=dict(epoch=epoch,seconds=time.perf_counter()-started,head_losses=(sums/steps).tolist(),
                auxiliary_steps=len(selected),sampled_windows=windows,masked_frames=masked,
                sampled_quality_frames={s:dict(zip([*QUALITIES,'N'],map(int,v))) for s,v in exposure.items()},**metrics)
            history.append(row);print(json.dumps(dict(arm=arm,**row)),flush=True)
            (directory/'history.json').write_text(json.dumps(history,indent=2))
            if metrics['frame_accuracy']>best:
                best=metrics['frame_accuracy'];torch.save(dict(state_dict=model.state_dict(),epoch=epoch,
                    width=cfg['width'],classes=CLASSES,model='multi_input_frame_TCN'),directory/'best.pt')
        checkpoint=torch.load(directory/'best.pt',map_location=device,weights_only=True)
        model.load_state_dict(checkpoint['state_dict']);predictions=directory/'predictions';predictions.mkdir()
        metrics=evaluate(model,val_songs,device,predictions)
        metrics.update(best_epoch=checkpoint['epoch'],smoke=smoke,evaluation_partition=record['evaluation_partition'],
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)
        (directory/'metrics.json').write_text(json.dumps(metrics,indent=2))
        pool_events(predictions,val_events,directory);outcomes[arm]=metrics
    (output/'comparison.json').write_text(json.dumps(dict(seed=cfg['seed'],smoke=smoke,results=outcomes),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/p1-jaah-pilot.json'))
    p.add_argument('--cache',type=Path,default=Path('data/processed/p1'))
    p.add_argument('--aux-cache',type=Path,default=Path('data/processed/p2-jaah'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.config,a.cache,a.aux_cache,a.output,a.device,a.smoke)
