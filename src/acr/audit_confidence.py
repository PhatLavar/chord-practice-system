"""Development-only event-level confidence audit of the frozen F/E primary pair."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from data_e import load_events, partitions
from diagnose_f import probabilities, decode_components
from p1_data import QUALITIES, TRIADS, EXTENSIONS
from teacher_f import targets

HEADS = ('gate', 'root', 'triad', 'extension')
VALUES = [('chord', 'N'), tuple(map(str, range(12))), TRIADS, EXTENSIONS]
THRESHOLDS = (0., .5, .7, .8, .9, .95, .99)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tally(target, prediction, confidence, selected, eligible, songs):
    selected = selected & eligible
    count = int(selected.sum())
    correct = int((selected & (target == prediction)).sum())
    total = int(eligible.sum())
    return dict(eligible_events=total, accepted=count, correct=correct,
                precision=correct/count if count else None,
                coverage=count/total if total else None,
                correct_coverage=correct/total if total else None,
                mean_confidence=float(confidence[selected].mean()) if count else None,
                accepted_songs=len(set(songs[selected])))


def decisions(f, e, head, method, threshold, operational):
    """Agreement confidence is the minimum of two scores, not a joint probability."""
    fp, ep = f[head].argmax(0), e[head].argmax(0)
    fc, ec = f[head].max(0), e[head].max(0)
    if method == 'F':
        pred, conf, mask = fp, fc, np.ones(len(fp), dtype=bool)
        gates = [f[0]]
    elif method == 'E':
        pred, conf, mask = ep, ec, np.ones(len(ep), dtype=bool)
        gates = [e[0]]
    else:
        pred, conf, mask = fp, np.minimum(fc, ec), fp == ep
        gates = [f[0], e[0]]
    mask = mask & (conf >= threshold)
    if operational and head:
        for gate in gates:
            mask &= (gate.argmax(0) == 0) & (gate[0] >= threshold)
    return pred, conf, mask


def load_pair(frun, erun, cache):
    meta = partitions(cache)
    fm = json.loads((frun/'run.json').read_text())
    em = json.loads((erun/'run.json').read_text())
    ids = sorted(i for i, p in meta['recordings'].items() if p == 'validation')
    for run in (fm, em):
        if run['evaluation_partition'] != 'validation' or run['evaluation_ids'] != ids or run['data'] != meta:
            raise ValueError('Run differs from frozen validation data')
        if run['config']['seed'] != 20261119:
            raise ValueError('Primary-pair audit requires original seed 20261119')
    if em['config'].get('balanced_probability') != .25:
        raise ValueError('Expected selected E25 policy')
    fref = Path('reports/p2/f-reference/events.csv')
    provenance = json.loads(fref.with_name('provenance.json').read_text())
    if sha(frun/'run.json') != provenance['F_run_record_sha256']:
        raise ValueError('F differs from frozen comparator')
    if json.loads((erun/'metrics.json').read_text())['F_reference_sha256'] != sha(fref):
        raise ValueError('E used a different F reference')
    songs, events, empty = load_events(cache, Path('data/raw/full.tar.xz'), ids)
    fr, er = read_rows(fref), read_rows(erun/'events.csv')
    if len(fr) != len(events) or len(er) != len(events):
        raise ValueError('Event count mismatch')
    for event, a, b in zip(events, fr, er):
        key = (event[0], event[1], event[2], event[-1])
        for row in (a, b):
            if (row['id'], float(row['start']), float(row['end']), int(row['reference'])) != key:
                raise ValueError('Event order, interval or target mismatch')
    pooled = [[] for _ in HEADS]
    hashes = {str(p): sha(p) for p in (frun/'run.json', erun/'run.json', erun/'event_logits.npz', erun/'events.csv', fref)}
    by_song = {}
    for event in events:
        by_song.setdefault(event[0], []).append(event)
    for rid, intervals in by_song.items():
        path = frun/'predictions'/f'{rid}.npz'
        hashes[str(path)] = sha(path)
        with np.load(path) as data:
            if not np.array_equal(data['times'], songs[rid]['times']) or not np.array_equal(data['reference'], songs[rid]['y']):
                raise ValueError('F timestamps or frame references mismatch')
            logits = [data[f'logits_{j}'] for j in range(4)]
            if not all(np.isfinite(v).all() and v.shape == (len(VALUES[j]), len(data['times'])) for j, v in enumerate(logits)):
                raise ValueError('Invalid F logits')
            probs = [probabilities(v) for v in logits]
            for _, _, _, lo, hi, _ in intervals:
                for j in range(4):
                    pooled[j].append(probs[j][:, lo:hi].mean(1))
    f = [np.stack(v, axis=1) for v in pooled]
    y = np.array([v[-1] for v in events])
    with np.load(erun/'event_logits.npz') as data:
        logits = [data[f'logits_{j}'] for j in range(4)]
        if not all(np.isfinite(v).all() and v.shape == (len(VALUES[j]), len(y)) for j, v in enumerate(logits)):
            raise ValueError('Invalid E logits')
        e = [probabilities(v) for v in logits]
        if not np.array_equal(data['reference'], y) or not np.array_equal(data['prediction'], decode_components(e)):
            raise ValueError('E logits/reference mismatch')
    for probs, rows in ((f, fr), (e, er)):
        if not np.array_equal(decode_components(probs), [int(r['prediction']) for r in rows]):
            raise ValueError('Complete predictions mismatch')
        for j, head in enumerate(HEADS):
            if not np.array_equal(probs[j].argmax(0), [int(r[head]) for r in rows]):
                raise ValueError('Component predictions mismatch')
    return f, e, y, np.array([v[0] for v in events]), dict(
        source_sha256=hashes, F_revision=fm['git_revision'], E_revision=em['git_revision'],
        manifest_sha256=meta['manifest_sha256'], events_without_frames=empty,
        seed=20261119, events=len(y), songs=len(ids))


def audit(frun, erun, cache, output):
    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    f, e, y, songs, summary = load_pair(frun, erun, cache)
    truth = [v.numpy() for v in targets(torch.from_numpy(y))]
    qualities = np.array(['N' if v == 0 else QUALITIES[(v-1) % len(QUALITIES)] for v in y])
    rare = [r['quality'] for r in read_rows(Path('reports/p0/qualities.csv')) if r['rare'] == 'True']
    curves, classes, quality_rows, bins = [], [], [], []
    for j, head in enumerate(HEADS):
        for method in ('F', 'E', 'agreement'):
            for scope in ('conditional', 'operational'):
                eligible = truth[j] >= 0 if scope == 'conditional' else np.ones(len(y), dtype=bool)
                for threshold in THRESHOLDS:
                    pred, conf, mask = decisions(f, e, j, method, threshold, scope == 'operational')
                    prefix = dict(head=head, method=method, scope=scope, threshold=threshold)
                    curves.append(dict(**prefix, **tally(truth[j], pred, conf, mask, eligible, songs)))
                    if scope == 'operational':
                        for value, label in enumerate(VALUES[j]):
                            stats = tally(truth[j], pred, conf, mask & (pred == value), eligible, songs)
                            support = int((truth[j] == value).sum())
                            classes.append(dict(**prefix, value=label, support=support,
                                                accepted_recall=stats['correct']/support if support else None, **stats))
                        for quality in ['N', *QUALITIES, 'ALL_RARE']:
                            subset = np.isin(qualities, rare) if quality == 'ALL_RARE' else qualities == quality
                            quality_rows.append(dict(**prefix, true_quality=quality,
                                **tally(truth[j], pred, conf, mask, subset, songs)))
            # Fixed-width descriptive reliability bins, before gate filtering.
            pred, conf, mask = decisions(f, e, j, method, 0., False)
            eligible = truth[j] >= 0
            for index in range(10):
                lo, hi = index/10, (index+1)/10
                selected = mask & (conf >= lo) & ((conf < hi) if index < 9 else (conf <= hi))
                bins.append(dict(head=head, method=method, lower=lo, upper=hi,
                    **tally(truth[j], pred, conf, selected, eligible, songs)))
    fp, ep = decode_components(f), decode_components(e)
    agreement = fp == ep
    summary.update(protocol='Validation development audit on gold intervals; F mean frame probabilities, E event softmax. No fitted thresholds.',
        thresholds=THRESHOLDS, rare_qualities=rare,
        whole_label_agreement=tally(y, fp, np.zeros(len(y)), agreement, np.ones(len(y), dtype=bool), songs),
        whole_label_rare_agreement=tally(y, fp, np.zeros(len(y)), agreement, np.isin(qualities, rare), songs))
    for key in ('whole_label_agreement', 'whole_label_rare_agreement'):
        summary[key].pop('mean_confidence')
    output.mkdir(parents=True)
    for name, rows in [('curves', curves), ('predicted_classes', classes), ('true_qualities', quality_rows), ('reliability_bins', bins)]:
        write_rows(output/f'{name}.csv', rows)
    summary['audit_script_sha256'] = sha(Path(__file__))
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--f-run', type=Path, default=Path('runs/p1-balanced-kaggle'))
    parser.add_argument('--e-run', type=Path, default=Path('runs/p2-mixture25-kaggle'))
    parser.add_argument('--cache', type=Path, default=Path('data/processed/p1'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.f_run, args.e_run, args.cache, args.output)
