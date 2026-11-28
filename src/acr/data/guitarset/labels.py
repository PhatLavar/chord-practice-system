"""Conservative GuitarSet label equivalence and progression-disjoint manifests.

This produces supervision proposals only; existing training targets are unchanged.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import zipfile

import mir_eval
import numpy as np
from acr.data.guitarset.audit import MD5, ERRATA
from acr.vocabulary import COMPONENTS, mapping
from acr.data.billboard.corpus import table
from acr.vocabulary import QUALITIES, TRIADS, EXTENSIONS


def spelling(symbol):
    """Relative pitch set, preserving additions/omissions; bass is a separate head.

    Do not use encode(): it inserts bass into the bitmap, even when root is omitted.
    Extended degrees are folded by pitch class, never discarded above the octave.
    """
    root, quality, degrees, bass = mir_eval.chord.split(symbol, reduce_extended_chords=True)
    bits = mir_eval.chord.quality_to_bitmap(quality).copy()
    bits[0] = 1
    for degree in degrees:
        bits += mir_eval.chord.scale_degree_to_bitmap(degree, modulo=True)
    return mir_eval.chord.pitch_class_to_semitone(root), tuple(map(int,bits>0))


TEMPLATES = {spelling('C:'+q)[1]:q for q in QUALITIES}
assert len(TEMPLATES)==len(QUALITIES)


def map_symbol(symbol):
    if symbol=='X':
        return dict(status='unknown',quality='X',complete=-100,gate=-100,root=-100,triad=-100,extension=-100)
    if symbol=='N':
        return dict(status='no_chord',quality='N',complete=0,gate=1,root=-100,triad=-100,extension=-100)
    root,bits=spelling(symbol)  # Malformed labels must fail, not silently become N/OOV.
    quality=TEMPLATES.get(bits)
    result=dict(status='root_gate_only',quality='OOV',complete=-100,gate=0,root=root,triad=-100,extension=-100)
    if quality is not None:
        triad,extension=COMPONENTS[quality]
        result.update(status='exact_pitch_set',quality=quality,complete=1+root*14+QUALITIES.index(quality),
                      triad=TRIADS.index(triad),extension=EXTENSIONS.index(extension))
    return result


def progression_signature(events):
    """Transpose-invariant ordered chord sequence, ignoring tempo and repeated holds."""
    sequence=[];base=None
    for event in events:
        if event['value'] in {'X','N'}:raise ValueError('Unexpected instructed no-chord/unknown')
        root,bits=spelling(event['value'])
        if base is None:base=root
        token=((root-base)%12,bits)
        if not sequence or sequence[-1]!=token:sequence.append(token)
    if not sequence:raise ValueError('Empty instructed sequence')
    return hashlib.sha256(json.dumps(sequence).encode()).hexdigest()


def fold_role(group,heldout):
    if group==heldout:return 'test'
    return 'validation' if group==(heldout+1)%3 else 'train'


def run(raw, output):
    if output.exists():raise FileExistsError('Use a fresh report directory')
    with raw.open('rb') as f:
        if hashlib.file_digest(f,'md5').hexdigest()!=MD5['annotation.zip']:raise ValueError('Wrong annotation release')
    records=[];events=[];symbols=Counter();seconds=Counter();groups={}
    with zipfile.ZipFile(raw) as z:
        for name in sorted(z.namelist()):
            if not name.endswith('.jams') or name.startswith('__MACOSX'):continue
            rid=Path(name).stem;doc=json.loads(z.read(name))
            chords=[a for a in doc['annotations'] if a['namespace']=='chord']
            performed=[a for a in chords if 'Chord sheet-informed' in a['annotation_metadata']['annotation_rules']]
            instructed=[a for a in chords if not a['annotation_metadata']['annotation_rules']]
            if len(performed)!=1 or len(instructed)!=1:raise ValueError('Ambiguous annotation layers')
            signature=progression_signature(instructed[0]['data'])
            player,sheet,mode=rid.split('_');code=sheet.split('-')[0]
            group=int(code[-1])-1
            if group not in range(3):raise ValueError('Unexpected progression code')
            previous=groups.setdefault(group,signature)
            if previous!=signature:raise ValueError('Progression code disagrees with label sequence')
            excluded=rid in ERRATA
            records.append((rid,player,code,mode,group,signature,excluded))
            for e in performed[0]['data']:
                mapped=map_symbol(e['value']);duration=float(e['duration'])
                if duration<=0 or not np.isfinite([e['time'],duration]).all():raise ValueError('Invalid interval')
                events.append([rid,e['time'],e['time']+duration,e['value'],excluded,*mapped.values()])
                if not excluded:
                    key=(e['value'],mapping(e['value'])[0],mapped['quality'],mapped['status'])
                    symbols[key]+=1;seconds[key]+=duration
    if len(records)!=360 or len(set(groups.values()))!=3:raise ValueError('Unexpected group inventory')
    output.mkdir(parents=True)
    table(output/'symbol_mapping.csv','symbol old_quality quality status events seconds',
          [list(k)+[symbols[k],seconds[k]] for k in sorted(symbols)])
    table(output/'event_targets.csv','id start end symbol excluded status quality complete gate root triad extension',events)
    folds=[]
    for heldout in range(3):
        for rid,player,code,mode,group,signature,excluded in records:
            folds.append([heldout,rid,player,code,mode,group,signature,'excluded' if excluded else fold_role(group,heldout)])
    table(output/'folds.csv','fold id player style_progression mode progression signature partition',folds)
    fold_support=defaultdict(lambda:[0,0.])
    for heldout in range(3):
        roles={r[1]:r[-1] for r in folds if r[0]==heldout}
        for event in events:
            rid,start,end,_,excluded,status,quality,*_=event
            if excluded:continue
            bucket=fold_support[(heldout,roles[rid],quality)]
            bucket[0]+=1;bucket[1]+=end-start
    table(output/'fold_quality_support.csv','fold partition quality events seconds',
          [list(k)+v for k,v in sorted(fold_support.items())])
    totals=defaultdict(float)
    for k,duration in seconds.items():totals[k[3]]+=duration
    all_seconds=sum(totals.values())
    summary=dict(annotation_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),mir_eval=mir_eval.__version__,
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 candidate_recordings=358,excluded=sorted(ERRATA),groups=groups,
                 performed_seconds=all_seconds,status_seconds=dict(totals),
                 complete_supported_fraction=(totals['exact_pitch_set']+totals['no_chord'])/all_seconds,
                 old_complete_supported_fraction=sum(v for k,v in seconds.items() if k[1]!='OOV')/all_seconds,
                 protocol='Three outer progression folds; cyclic next group validation, remaining group train. Same players across partitions; not unseen-player evaluation. No model training.',
                 partial_policy='OOV receives annotated root/gate proposals only; triad/extension/complete masked. Proposals not enabled in trainers. Bass ignored consistently with existing task.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,default=Path('data/raw/guitarset/annotation.zip'))
    p.add_argument('--output',type=Path,default=Path('reports/p2/guitarset-labels'))
    a=p.parse_args();run(a.raw,a.output)
