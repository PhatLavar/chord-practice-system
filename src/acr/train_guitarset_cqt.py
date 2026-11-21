"""Train fresh CQT-TCN F/E on one GuitarSet development fold; never evaluate outer test."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader
from data_e import Events, collate_events
from evaluate_intervals import score
from guitarset_labels import map_symbol
from p1_data import QUALITIES
from prepare_guitarset_cqt import load_cache, digest
from teacher_f import FrameTeacher, decode, losses
from teacher_e import EventTeacher
from train_f import Windows

NOTES=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


def chord_label(value):
    return 'N' if value==0 else NOTES[(value-1)//14]+':'+QUALITIES[(value-1)%14]


def event_list(songs, references):
    events=[]
    for rid in sorted(songs):
        for e in references[rid]['events']:
            start,end=e['time'],e['time']+e['duration']
            lo,hi=np.searchsorted(songs[rid]['times'],[start,end],side='left')
            if hi==lo:raise ValueError('Gold interval has no CQT frames; revise event protocol explicitly')
            events.append((rid,start,end,int(lo),int(hi),map_symbol(e['value'])['complete']))
    return events


def timeline(events, duration, labels, fill):
    intervals=[];result=[];previous=0.
    for e,label in zip(events,labels):
        start,end=float(e['time']),float(e['time']+e['duration'])
        # GuitarSet decimal endpoints can exceed WAV duration by <1 original sample.
        if start<previous-1e-8 or end>duration+1/44100:raise ValueError('Invalid reference coverage')
        end=min(end,duration)
        if end<=start:raise ValueError('Empty reference interval after clipping')
        if start>previous+1e-8:intervals.append([previous,start]);result.append(fill)
        intervals.append([start,end]);result.append(label);previous=end
    if previous<duration:intervals.append([previous,duration]);result.append(fill)
    return intervals,result


@torch.no_grad()
def evaluate(model, teacher, songs, references, device, batch_size, output=None):
    model.eval();events=event_list(songs,references);predicted={};probabilities={}
    if teacher=='F':
        for rid,s in songs.items():
            x=torch.from_numpy(s['x'].T.copy()).unsqueeze(0).to(device)
            # Match the zero-normalized-input context used by the training windows.
            x=torch.nn.functional.pad(x,(model.context,model.context))
            heads=[h[:,:,model.context:model.context+len(s['times'])] for h in model(x)]
            predicted[rid]=decode(heads)[0].cpu().numpy()
            probabilities[rid]=[h.softmax(1)[0].cpu().numpy() for h in heads]
            if output:
                np.savez_compressed(output/(rid+'.npz'),prediction=predicted[rid],times=s['times'],
                                    **{f'logits_{j}':h[0].cpu().numpy() for j,h in enumerate(heads)})
    else:
        dataset=Events(songs,events,context=model.context)
        parts=[[] for _ in range(4)]
        for x,mask,_ in DataLoader(dataset,batch_size=batch_size,shuffle=False,collate_fn=collate_events):
            for dest,h in zip(parts,model(x.to(device),mask.to(device))):dest.append(h.cpu())
        heads=[torch.cat(p) for p in parts];labels=decode(heads).reshape(-1).numpy()
        if output:np.savez_compressed(output/'event_logits.npz',prediction=labels,**{f'logits_{j}':h[:,:,0].numpy() for j,h in enumerate(heads)})
        for e,label in zip(events,labels):predicted.setdefault(e[0],[]).append(int(label))
    totals={};per_song={};correct=count=0;quality=defaultdict(lambda:dict(events=0,correct=0,predicted=0))
    for rid,s in songs.items():
        reference=references[rid];raw=reference['events'];duration=reference['duration']
        ref_i,ref_l=timeline(raw,duration,[e['value'] for e in raw],'X')
        if teacher=='F':
            times=s['times'];est_i=np.stack([times,np.r_[times[1:],duration]],axis=1)
            est_l=[chord_label(int(v)) for v in predicted[rid]]
            event_predictions=[]
            for e in raw:
                lo,hi=np.searchsorted(times,[e['time'],e['time']+e['duration']])
                pooled=[torch.from_numpy(p[:,lo:hi].mean(1)).clamp_min(1e-30).log()[None,:,None] for p in probabilities[rid]]
                event_predictions.append(int(decode(pooled).item()))
        else:
            event_predictions=predicted[rid]
            est_i,est_l=timeline(raw,duration,[chord_label(v) for v in event_predictions],'N')
        result=score(ref_i,ref_l,est_i,est_l);per_song[rid]=result
        if output:
            # Preserve float64 endpoints, including tiny positive trailing gaps.
            (output/(rid+'.lab')).write_text(''.join(f'{a:.17g} {b:.17g} {label}\n' for (a,b),label in zip(est_i,est_l)))
        for name,values in result['metrics'].items():
            bucket=totals.setdefault(name,dict(correct_seconds=0.,eligible_seconds=0.,excluded_seconds=0.))
            for k in bucket:bucket[k]+=values[k]
        for e,pred in zip(raw,event_predictions):
            target=map_symbol(e['value'])['complete']
            if target<0:continue
            correct+=int(pred==target);count+=1
            q='N' if target==0 else QUALITIES[(target-1)%14]
            quality[q]['events']+=1;quality[q]['correct']+=int(pred==target)
            pq='N' if pred==0 else QUALITIES[(pred-1)%14];quality[pq]['predicted']+=1
    for values in totals.values():
        values['score']=values['correct_seconds']/values['eligible_seconds'] if values['eligible_seconds'] else None
    result=dict(teacher=teacher,metrics=totals,supported_oracle_event_accuracy=correct/count if count else None,
                supported_events=count,quality_supported_scope=dict(quality),
                protocol='Full audio duration with raw performed labels; gaps X. F predicted frame intervals; E gold intervals. Event accuracy restricted to complete-supported labels.')
    if output:(output/'per_song.json').write_text(json.dumps(per_song,indent=2)+'\n')
    return result


def train(cache, config, output, device, smoke=False):
    if output.exists():raise FileExistsError('Use a fresh run directory')
    cfg=json.loads(config.read_text());songs,refs,meta=load_cache(cache,cfg,smoke)
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('Enable GPU')
    torch.set_num_threads(4)
    train_songs={i:s for i,s in songs.items() if meta['recordings'][i]=='train'}
    validation={i:s for i,s in songs.items() if meta['recordings'][i]=='validation'}
    output.mkdir(parents=True);summary={}
    (output/'validation_references.json').write_text(json.dumps({i:refs[i] for i in validation},indent=2)+'\n')
    for teacher,cls in [('F',FrameTeacher),('E',EventTeacher)]:
        random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
        model=cls(width=cfg['width'],input_channels=cfg['n_bins']).to(device)
        if teacher=='F':
            dataset=Windows(train_songs,cfg['window'],model.context);collate=None
        else:
            supported=[e for e in event_list(train_songs,refs) if e[-1]>=0]
            dataset=Events(train_songs,supported,context=model.context);collate=collate_events
        loader=DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=True,collate_fn=collate)
        if not len(loader):raise ValueError('No training batches')
        directory=output/teacher;directory.mkdir()
        epochs=2 if smoke else cfg['epochs'];steps=min(2,len(loader)) if smoke else len(loader)
        record=dict(config=cfg,teacher=teacher,smoke=smoke,epochs=epochs,steps_per_epoch=steps,
                    train_ids=sorted(train_songs),validation_ids=sorted(validation),cache_metadata=meta,
                    torch=torch.__version__,device=device,
                    git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    source_sha256={p.as_posix():digest(p) for p in Path('src/acr').glob('*.py')},
                    protocol='Fresh CQT-TCN; natural sampling; complete labels only; select validation duration-weighted Tetrads; no outer test.')
        (directory/'run.json').write_text(json.dumps(record,indent=2)+'\n')
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
        best=-1;history=[]
        for epoch in range(1,epochs+1):
            model.train();sums=np.zeros(4)
            for step,batch in enumerate(loader):
                if step>=steps:break
                if teacher=='F':
                    x,y=batch;heads=[h[:,:,model.context:model.context+cfg['window']] for h in model(x.to(device))]
                else:
                    x,mask,y=batch;heads=model(x.to(device),mask.to(device))
                terms=losses(heads,y.to(device));loss=sum(terms)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                sums+=[float(t.detach()) for t in terms]
            metrics=evaluate(model,teacher,validation,refs,device,cfg['batch_size'])
            value=metrics['metrics']['tetrads']['score']
            if value is None:raise ValueError('No eligible validation duration')
            row=dict(epoch=epoch,head_losses=(sums/steps).tolist(),validation_tetrads=value)
            if cfg.get('track_training_metrics',False):
                fitted=evaluate(model,teacher,train_songs,refs,device,cfg['batch_size'])
                row['training_metrics']=fitted
                row['validation_metrics']=metrics
            history.append(row)
            progress=dict(teacher=teacher,epoch=epoch,head_losses=row['head_losses'],validation_tetrads=value)
            if 'training_metrics' in row:progress['training_tetrads']=row['training_metrics']['metrics']['tetrads']['score']
            print(json.dumps(progress),flush=True)
            (directory/'history.json').write_text(json.dumps(history,indent=2)+'\n')
            if value>best:
                best=value;torch.save(dict(state_dict=model.state_dict(),epoch=epoch,config=cfg,teacher=teacher),directory/'best.pt')
        checkpoint=torch.load(directory/'best.pt',map_location=device,weights_only=True)
        model.load_state_dict(checkpoint['state_dict']);predictions=directory/'predictions';predictions.mkdir()
        metrics=evaluate(model,teacher,validation,refs,device,cfg['batch_size'],predictions)
        metrics.update(best_epoch=checkpoint['epoch'],evaluation_partition='validation_subset_smoke' if smoke else 'validation',smoke=smoke)
        (directory/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');summary[teacher]=metrics
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/p2-guitarset-cqt.json'))
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();train(a.cache,a.config,a.output,a.device,a.smoke)
