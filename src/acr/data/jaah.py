"""Pin and audit JAAH frame chroma for an auxiliary-training-only pilot."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import quote
from urllib.request import urlopen
import zipfile

import numpy as np
from acr.data.billboard.corpus import table
from acr.vocabulary import encode, QUALITIES, CLASSES

REVISION = '6fec3f6dd1d60c21cf9af855ea173be38e6d7243'
BASE = f'https://raw.githubusercontent.com/MTG/JAAH/{REVISION}/'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_title(title):
    # Conservative composition exclusion includes alternate performances/artists.
    text = unicodedata.normalize('NFKD', title).casefold()
    return re.sub(r'[^a-z0-9]', '', text)


def external_label(symbol):
    # Harte's bare root is a major triad. Other unsupported forms remain masked.
    if re.fullmatch(r'[A-G][b#]*', symbol):
        symbol += ':maj'
    return encode(symbol)


def parse_intervals(text):
    result = []
    previous = 0.
    for line in text.splitlines():
        if not line.strip():
            continue
        start, end, symbol = line.split(maxsplit=2)
        start, end = float(start), float(end)
        if not np.isfinite([start, end]).all() or start < previous or end <= start:
            raise ValueError('Invalid/overlapping JAAH interval')
        previous = end
        label = external_label(symbol)
        if result and label >= 0 and result[-1][2] == label and result[-1][1] == start:
            result[-1] = (result[-1][0], end, label)
        else:
            result.append((start, end, label))
    return result


def download(raw):
    def fetch(relative):
        path = raw/relative
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(BASE+quote(relative, safe='/'), timeout=60) as response:
            content = response.read()
        temporary = path.with_suffix(path.suffix+'.part')
        temporary.write_bytes(content)
        temporary.replace(path)
    fetch('features/db.json'); fetch('labs.zip')
    db = json.loads((raw/'features/db.json').read_text())
    paths = []
    for row in db:
        name = row['nnls_chroma']['path']
        if Path(name).name != name:
            raise ValueError('Unsafe feature filename')
        paths.extend(['features/'+name, 'annotations/'+name.removesuffix('.csv.gz')+'.json'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, paths))
    print(f'Downloaded pinned JAAH files for {len(db)} recordings', flush=True)


def prepare(raw, output, report):
    if output.exists() or report.exists():
        raise FileExistsError('Use fresh cache/report directories')
    db = json.loads((raw/'features/db.json').read_text())
    with Path('data/raw/index.csv').open(encoding='utf-8-sig') as stream:
        billboard_titles = {normalize_title(r['title']) for r in csv.DictReader(stream) if r['title']}
    manifest = []; events = []; songs = {}; source_hashes = {}
    counts = Counter(); raw_counts = Counter(); quality_songs = defaultdict(set)
    quality_compositions = defaultdict(set)
    seen_features = set(); excluded = []; empty = 0
    with zipfile.ZipFile(raw/'labs.zip') as archive:
        if archive.testzip() is not None:
            raise ValueError('Corrupt LAB archive')
        for item in sorted(db, key=lambda r: r['nnls_chroma']['path']):
            feature = item['nnls_chroma']; name = feature['path'].removesuffix('.csv.gz')
            if feature['sample_rate'] != 44100 or feature['step_size'] != 2048:
                raise ValueError('Unexpected JAAH feature clock')
            annotation_path = raw/'annotations'/f'{name}.json'
            info = json.loads(annotation_path.read_text())
            feature_path = raw/'features'/feature['path']
            feature_hash = digest(feature_path)
            source_hashes[annotation_path.relative_to(raw).as_posix()] = digest(annotation_path)
            source_hashes[feature_path.relative_to(raw).as_posix()] = feature_hash
            reason = ('billboard_title_overlap' if normalize_title(info['title']) in billboard_titles else
                      'duplicate_feature' if feature_hash in seen_features else '')
            seen_features.add(feature_hash)
            manifest.append([name, info['title'], info['artist'], normalize_title(info['title']), reason or 'auxiliary_train'])
            if reason:
                excluded.append(dict(id=name, reason=reason)); continue
            text = archive.read(f'labs/{name}.lab').decode()
            intervals = parse_intervals(text)
            with gzip.open(feature_path, 'rt') as stream:
                values = np.loadtxt(stream, delimiter=',')
            if values.ndim != 2 or values.shape[1] != 13 or not np.isfinite(values).all():
                raise ValueError(f'Invalid 12-bin frame chroma: {name}')
            times, x = values[:, 0], values[:, 1:].astype(np.float32)
            if len(times) < 2 or times[0] < 0 or not np.allclose(np.diff(times), 2048/44100, atol=1e-7, rtol=0) or (x < 0).any():
                raise ValueError(f'Invalid feature values/clock: {name}')
            if intervals[-1][1] > times[-1]+.25 or abs(len(times)*2048/44100-float(info['duration'])) > .5:
                manifest[-1][-1] = 'duration_mismatch'
                excluded.append(dict(id=name, reason='duration_mismatch', feature_end=float(times[-1]),
                    label_end=intervals[-1][1], metadata_duration=float(info['duration'])))
                continue
            for line in text.splitlines():
                if line.strip():
                    raw_counts[line.split(maxsplit=2)[2]] += 1
            x /= np.maximum(x.sum(1, keepdims=True), 1e-8)
            y = np.full(len(times), -100, dtype=np.int64)
            rid = 'jaah-'+name
            for start, end, label in intervals:
                lo, hi = np.searchsorted(times, [start, end], side='left')
                y[lo:hi] = label
                if label < 0:
                    counts['OOV'] += 1
                    quality_songs['OOV'].add(rid)
                    quality_compositions['OOV'].add(normalize_title(info['title']))
                    continue
                if hi == lo:
                    empty += 1; continue
                events.append([rid, start, end, int(lo), int(hi), label])
                q = 'N' if label == 0 else QUALITIES[(label-1) % 14]
                counts[q] += 1; quality_songs[q].add(rid)
                quality_compositions[q].add(normalize_title(info['title']))
            songs[rid] = dict(x=x, y=y, times=times)
    if not songs or not events:
        raise ValueError('No usable JAAH data')
    for relative in ['features/db.json', 'labs.zip']:
        source_hashes[relative] = digest(raw/relative)
    output.mkdir(parents=True); report.mkdir(parents=True)
    cache_hashes = {}
    for rid, song in songs.items():
        path = output/f'{rid}.npz'; np.savez_compressed(path, **song); cache_hashes[path.name] = digest(path)
    (output/'events.json').write_text(json.dumps(events))
    cache_hashes['events.json'] = digest(output/'events.json')
    metadata = dict(revision=REVISION, base_url=BASE, classes=CLASSES,
        partition='auxiliary_train_only', recordings=sorted(songs), events=len(events),
        excluded=excluded, empty_events=empty, channels=12, step_seconds=2048/44100,
        normalization='Per-frame L1; no statistics fitted across songs',
        mapping='Bare major root canonicalization only; unsupported symbols masked',
        overlap_check='Conservative exact normalized title against ALL Billboard index entries; no audio fingerprint claim',
        source_sha256=source_hashes, cache_sha256=cache_hashes,
        billboard_index_sha256=digest(Path('data/raw/index.csv')),
        preparation_script_sha256=digest(Path(__file__)))
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2))
    (report/'summary.json').write_text(json.dumps(metadata, indent=2))
    table(report/'inventory.csv', 'id title artist composition status', manifest)
    table(report/'qualities.csv', 'quality events recordings compositions',
          [[q, counts[q], len(quality_songs[q]), len(quality_compositions[q])] for q in [*QUALITIES, 'N', 'OOV']])
    table(report/'symbols.csv', 'symbol events', sorted(raw_counts.items()))
    print(json.dumps(dict(recordings=len(songs), events=len(events), counts=counts, excluded=excluded), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, default=Path('data/raw/jaah'))
    p.add_argument('--output', type=Path, default=Path('data/processed/p2-jaah'))
    p.add_argument('--report', type=Path, default=Path('reports/p2/jaah-inventory'))
    p.add_argument('--download', action='store_true')
    a = p.parse_args()
    if a.download: download(a.raw)
    prepare(a.raw, a.output, a.report)
