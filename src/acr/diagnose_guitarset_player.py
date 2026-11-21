"""Audit saved player-held-out F/E components on supported gold events."""
import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import soundfile as sf
import torch

from guitarset_labels import map_symbol
from p1_data import QUALITIES, TRIADS, EXTENSIONS
from teacher_f import decode, targets
from train_guitarset_cqt import chord_label


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run():
    root=Path('runs/p2-guitarset-cqt-player')
    out=Path('reports/p2/guitarset-cqt/player-diagnostics');out.mkdir(exist_ok=False)
    refs=json.loads((root/'validation_references.json').read_text())
    labels=[['chord','N'],list(range(12)),TRIADS,EXTENSIONS]
    records=[];summary={};matrices=[]
    for teacher in ['F','E']:
        offset=0;truth=[];heads=[[],[],[],[]];predictions=[]
        if teacher=='E':
            with np.load(root/teacher/'predictions/event_logits.npz') as data:
                allprob=[torch.from_numpy(data[f'logits_{j}']).softmax(1) for j in range(4)]
        for rid,ref in sorted(refs.items()):
            if teacher=='F':
                with np.load(root/teacher/'predictions'/(rid+'.npz')) as data:
                    times=data['times'];frame=[torch.from_numpy(data[f'logits_{j}']).softmax(0) for j in range(4)]
            for e in ref['events']:
                if teacher=='F':
                    lo,hi=np.searchsorted(times,[e['time'],e['time']+e['duration']]);assert hi>lo
                    prob=[p[:,lo:hi].mean(1) for p in frame]
                else:prob=[p[offset] for p in allprob]
                offset+=1
                target=map_symbol(e['value'])['complete']
                if target<=0:continue
                predicted=int(decode([p.clamp_min(1e-30).log()[None,:,None] for p in prob]).item())
                truth.append(target);predictions.append(predicted)
                components=[int(p.argmax()) for p in prob]
                for dest,c in zip(heads,components):dest.append(c)
                records.append(dict(teacher=teacher,recording=rid,start=e['time'],end=e['time']+e['duration'],
                    reference=e['value'],target=chord_label(target),prediction=chord_label(predicted),
                    root_confidence=float(prob[1].max()),triad_confidence=float(prob[2].max()),
                    extension_confidence=float(prob[3].max()),extension_prediction=EXTENSIONS[components[3]]))
        target_heads=targets(torch.tensor(truth));counts={}
        for j,name in enumerate(['gate','root','triad','extension']):
            actual=target_heads[j].numpy();pred=np.array(heads[j]);matrix=np.zeros((len(labels[j]),len(labels[j])),dtype=int)
            np.add.at(matrix,(actual,pred),1)
            assert int(matrix.sum())==len(truth)
            for a,row in enumerate(matrix):
                for b,count in enumerate(row):matrices.append(dict(teacher=teacher,head=name,reference=labels[j][a],prediction=labels[j][b],events=int(count)))
            counts[name]=dict(correct=int((actual==pred).sum()),events=len(actual))
        actual_ext=target_heads[3].numpy();actual_root=target_heads[1].numpy();actual_triad=target_heads[2].numpy()
        pred=np.array(predictions);correct_root=np.array(heads[1])==actual_root;correct_triad=np.array(heads[2])==actual_triad;correct_ext=np.array(heads[3])==actual_ext
        extended=actual_ext!=EXTENSIONS.index('none')
        saved=json.loads((root/teacher/'metrics.json').read_text())
        assert len(truth)==saved['supported_events']
        assert abs(float(np.mean(pred==truth))-saved['supported_oracle_event_accuracy'])<1e-12
        summary[teacher]=dict(components=counts,supported_events=len(truth),extended_events=int(extended.sum()),
            extended_predicted_none=int((extended & (np.array(heads[3])==EXTENSIONS.index('none'))).sum()),
            extension_correct_but_complete_wrong=int((correct_ext & (pred!=truth)).sum()),
            root_and_triad_correct_extension_wrong=int((correct_root & correct_triad & ~correct_ext).sum()),
            raw_all_heads_correct=int((correct_root & correct_triad & correct_ext).sum()))
    write_csv(out/'confusion.csv',matrices);write_csv(out/'events.csv',records)
    # Paired examples are selected deterministically, before any listening judgment.
    frows=[r for r in records if r['teacher']=='F'];erows=[r for r in records if r['teacher']=='E']
    assert [(r['recording'],r['start']) for r in frows]==[(r['recording'],r['start']) for r in erows]
    selected=[]
    for quality in ['7','maj7','min7','maj6','min6']:
        candidates=[(f,e) for f,e in zip(frows,erows) if f['target'].split(':')[1]==quality and f['prediction']!=f['target'] and e['prediction']!=e['target']]
        if candidates:
            f,e=candidates[0];selected.append(dict(recording=f['recording'],start=f['start'],end=f['end'],reference=f['reference'],F=f['prediction'],E=e['prediction']))
    clips=Path('runs/p2-guitarset-player-review');clips.mkdir(exist_ok=False)
    with zipfile.ZipFile('data/raw/guitarset/audio_mono-mic.zip') as z:
        names={Path(n).stem.removesuffix('_mic'):n for n in z.namelist() if n.endswith('.wav') and not n.startswith('__MACOSX')}
        for i,row in enumerate(selected):
            audio,sr=sf.read(io.BytesIO(z.read(names[row['recording']])),dtype='float32')
            start=max(0,row['start']-1);end=min(len(audio)/sr,row['end']+1)
            path=clips/f'{i+1:02d}.wav';sf.write(path,audio[int(start*sr):int(end*sr)],sr,subtype='PCM_16')
            row.update(clip=path.as_posix(),clip_start=start,clip_end=end)
    write_csv(out/'review_candidates.csv',selected)
    summary['scope']='Supported chord gold events only; F mean frame probabilities, E event softmax. Raw head argmax confusion precedes legal quality decoder. Clips are review candidates, not independently verified chord judgments.'
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
