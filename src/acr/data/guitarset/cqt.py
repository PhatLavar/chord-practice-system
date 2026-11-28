"""Prepare train/validation CQT only for one frozen GuitarSet progression fold."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import librosa
import numpy as np
import soundfile as sf
from acr.data.guitarset.labels import map_symbol

FOLDS=Path('reports/p2/guitarset-labels/folds.csv')
INVENTORY=Path('reports/p2/guitarset-inventory/summary.json')


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def selected_records(fold, smoke=False, folds_path=FOLDS):
    with Path(folds_path).open() as f:rows=list(csv.DictReader(f))
    selected={r['id']:r['partition'] for r in rows if int(r['fold'])==fold and r['partition'] in ('train','validation')}
    if smoke:
        selected={rid:part for part in ('train','validation') for rid in sorted(i for i,p in selected.items() if p==part)[:2]}
    if set(selected.values())!={'train','validation'}:raise ValueError('Invalid fold')
    return selected


def event_list(songs, references):
    events=[]
    for rid in sorted(songs):
        for e in references[rid]['events']:
            start,end=e['time'],e['time']+e['duration']
            lo,hi=np.searchsorted(songs[rid]['times'],[start,end],side='left')
            if hi==lo:raise ValueError('Gold interval has no CQT frames; revise event protocol explicitly')
            events.append((rid,start,end,int(lo),int(hi),map_symbol(e['value'])['complete']))
    return events



def extract(audio, sample_rate, cfg):
    if audio.ndim!=1 or not np.isfinite(audio).all():raise ValueError('Expected finite mono audio')
    y=librosa.resample(audio,orig_sr=sample_rate,target_sr=cfg['sample_rate'],res_type='soxr_hq')
    cqt=librosa.cqt(y,sr=cfg['sample_rate'],hop_length=cfg['hop_length'],n_bins=cfg['n_bins'],
                    bins_per_octave=cfg['bins_per_octave'],fmin=cfg['fmin'],tuning=0.,
                    pad_mode='constant',res_type='soxr_hq')
    times=np.arange(cqt.shape[1],dtype=np.float64)*cfg['hop_length']/cfg['sample_rate']
    duration=len(audio)/sample_rate
    keep=times<duration
    features=np.log(np.abs(cqt[:,keep])+1e-6).T.astype(np.float32)
    if not np.isfinite(features).all():raise ValueError('Nonfinite CQT')
    return features,times[keep],duration


def frame_labels(times, reference):
    y=np.full(len(times),-100,dtype=np.int64)
    for e in reference:
        lo,hi=np.searchsorted(times,[e['time'],e['time']+e['duration']],side='left')
        y[lo:hi]=map_symbol(e['value'])['complete']
    return y


def normalization(songs, roles):
    count=0;total=0.;squares=0.
    for rid,song in songs.items():
        if roles[rid]!='train':continue
        x=song['x'].astype(np.float64)
        count+=x.size;total+=float(x.sum());squares+=float(np.square(x).sum())
    if not count:raise ValueError('No training samples for normalization')
    mean=total/count;std=float(np.sqrt(max(squares/count-mean*mean,0)))
    if std<1e-8:raise ValueError('Degenerate training normalization')
    return dict(mean=mean,std=std,values=count,fit_partition='train',train_ids=sorted(i for i,p in roles.items() if p=='train'))


def prepare(raw, output, config, smoke=False):
    if output.exists():raise FileExistsError('Use a fresh cache directory')
    cfg=json.loads(config.read_text());folds=Path(cfg.get('folds_path',FOLDS))
    roles=selected_records(cfg['fold'],smoke,folds)
    expected=json.loads(INVENTORY.read_text())['source_sha256']
    for name,sha in expected.items():
        if digest(raw/name)!=sha:raise ValueError(f'Wrong source: {name}')
    songs={};references={};audio_hashes={};output.mkdir(parents=True)
    with zipfile.ZipFile(raw/'annotation.zip') as labels,zipfile.ZipFile(raw/'audio_mono-mic.zip') as audio:
        labs={Path(n).stem:n for n in labels.namelist() if n.endswith('.jams') and not n.startswith('__MACOSX')}
        wavs={Path(n).stem.removesuffix('_mic'):n for n in audio.namelist() if n.endswith('.wav') and not n.startswith('__MACOSX')}
        for index,rid in enumerate(sorted(roles)):
            doc=json.loads(labels.read(labs[rid]))
            ref=[a for a in doc['annotations'] if a['namespace']=='chord' and 'Chord sheet-informed' in a['annotation_metadata']['annotation_rules']]
            if len(ref)!=1:raise ValueError('Ambiguous performed layer')
            data=audio.read(wavs[rid]);audio_hashes[rid]=hashlib.sha256(data).hexdigest()
            samples,sr=sf.read(io.BytesIO(data),dtype='float32')
            x,times,duration=extract(samples,sr,cfg)
            y=frame_labels(times,ref[0]['data']);songs[rid]=dict(x=x,y=y,times=times)
            references[rid]=dict(duration=duration,events=ref[0]['data'])
            np.savez_compressed(output/(rid+'.npz'),**songs[rid])
            if (index+1)%20==0 or smoke:print(f'Prepared {index+1}/{len(roles)}',flush=True)
    # Detect identical audio leaking across selected partitions.
    seen={}
    for rid,sha in audio_hashes.items():
        if sha in seen and roles[seen[sha]]!=roles[rid]:raise ValueError('Duplicate audio across partitions')
        seen[sha]=rid
    stats=normalization(songs,roles)
    (output/'references.json').write_text(json.dumps(references,indent=2)+'\n')
    metadata=dict(config=cfg,smoke=smoke,recordings=roles,normalization=stats,
                  source_sha256=expected,folds_sha256=digest(folds),audio_sha256=audio_hashes,
                  cache_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file()},
                  preparation_sha256=digest(Path(__file__)),label_mapper_sha256=digest(Path('src/acr/data/guitarset/labels.py')),
                  librosa=librosa.__version__,soundfile=sf.__version__,numpy=np.__version__,
                  protocol='BTC-inspired CQT parameters; full-recording centered CQT, timestamp labels, no augmentation. Raw log magnitude cached; scalar train-only normalization at loading. No outer-test audio/labels loaded.')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(recordings=len(roles),normalization=stats),indent=2))


def load_cache(cache, config, smoke):
    meta=json.loads((cache/'metadata.json').read_text())
    folds=Path(config.get('folds_path',FOLDS))
    if meta['config']!=config or meta['smoke']!=smoke or meta['folds_sha256']!=digest(folds):raise ValueError('Cache/config/fold mismatch')
    if meta['recordings']!=selected_records(config['fold'],smoke,folds):raise ValueError('Unexpected partition IDs')
    if meta['source_sha256']!=json.loads(INVENTORY.read_text())['source_sha256']:raise ValueError('Unpinned sources')
    for name,sha in meta['cache_sha256'].items():
        if digest(cache/name)!=sha:raise ValueError('Cache file changed')
    songs={}
    for rid in meta['recordings']:
        with np.load(cache/(rid+'.npz')) as d:songs[rid]={k:d[k] for k in ('x','y','times')}
    fitted=normalization(songs,meta['recordings']);saved=meta['normalization']
    if fitted['train_ids']!=saved['train_ids'] or fitted['values']!=saved['values'] or not np.allclose(
            [fitted['mean'],fitted['std']],[saved['mean'],saved['std']],rtol=1e-12,atol=1e-12):
        raise ValueError('Normalization provenance mismatch')
    for s in songs.values():
        s['x']=((s['x']-meta['normalization']['mean'])/meta['normalization']['std']).astype(np.float32)
    return songs,json.loads((cache/'references.json').read_text()),meta


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,default=Path('data/raw/guitarset'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/p2/p2-guitarset-cqt.json'))
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();prepare(a.raw,a.output,a.config,a.smoke)
