"""Matched E25 control versus auxiliary JAAH batches with a 12-bin input branch."""
import argparse
import csv
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data_e import Events, collate_events, load_events, partitions
from p1_data import CLASSES, QUALITIES
from prepare_jaah import digest, REVISION
from teacher_e import EventTeacher
from teacher_f import losses
from train_e import evaluate, compare


class MultiInputEventTeacher(EventTeacher):
    def __init__(self, width=48):
        super().__init__(width)
        # Do not perturb the original model's initialization/sampling RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.input12 = nn.Conv1d(12, width, 1)

    def encode_features(self, x):
        if x.shape[1] == 24:
            return super().encode_features(x)
        if x.shape[1] != 12:
            raise ValueError('Expected 12 or 24 input channels')
        h = F.relu(self.input12(x))
        for block in self.blocks:
            h = h + F.relu(block(h))
        return h


def load_aux(cache):
    meta = json.loads((cache/'metadata.json').read_text())
    pinned = json.loads(Path('reports/p2/jaah-inventory/summary.json').read_text())
    if meta['source_sha256'] != pinned['source_sha256'] or meta['recordings'] != pinned['recordings'] or meta['events'] != pinned['events']:
        raise ValueError('Auxiliary inputs differ from the audited inventory')
    if meta['revision'] != REVISION or meta['classes'] != CLASSES or meta['partition'] != 'auxiliary_train_only':
        raise ValueError('Unexpected auxiliary cache')
    if meta['billboard_index_sha256'] != digest(Path('data/raw/index.csv')):
        raise ValueError('Billboard overlap inventory changed')
    for name, expected in meta['cache_sha256'].items():
        if digest(cache/name) != expected:
            raise ValueError(f'Auxiliary cache changed: {name}')
    songs = {}
    for rid in meta['recordings']:
        with np.load(cache/f'{rid}.npz') as data:
            songs[rid] = {k: data[k] for k in ('x', 'y', 'times')}
    events = json.loads((cache/'events.json').read_text())
    return songs, events, meta


def replacement_steps(steps, fraction, seed, epoch):
    if not 0 <= fraction <= 1:
        raise ValueError('Invalid auxiliary fraction')
    rng = np.random.default_rng(seed+epoch)
    return set(map(int, rng.choice(steps, round(steps*fraction), replace=False)))


def run(config, cache, aux_cache, output, device, smoke=False):
    if output.exists():
        raise FileExistsError('Use a fresh experiment output directory')
    cfg = json.loads(config.read_text())
    if cfg['arms'] != ['billboard_control', 'jaah_mixed'] or cfg['seed'] != 20261119:
        raise ValueError('This pilot requires the prespecified matched primary seed/arms')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Enable a GPU')
    torch.set_num_threads(4)
    meta = partitions(cache)
    train_ids = sorted(i for i, p in meta['recordings'].items() if p == 'train_labeled')
    val_ids = sorted(i for i, p in meta['recordings'].items() if p == 'validation')
    if smoke:
        train_ids, val_ids = train_ids[:2], val_ids[:2]
    songs, events, _ = load_events(cache, Path('data/raw/full.tar.xz'), sorted(train_ids+val_ids))
    training = Events(songs, [e for e in events if e[0] in train_ids], balanced=True, balanced_probability=.25)
    validation = Events(songs, [e for e in events if e[0] in val_ids])
    aux_songs, aux_events, aux_meta = load_aux(aux_cache)
    auxiliary = Events(aux_songs, aux_events, balanced=True, balanced_probability=.25)
    reference = Path('reports/p2/f-reference/events.csv')
    if not smoke:
        with reference.open() as stream:
            frozen = [(r['id'], float(r['start']), float(r['end']), int(r['reference']))
                      for r in csv.DictReader(stream)]
        expected = [(e[0],e[1],e[2],e[-1]) for e in validation.events]
        if frozen != expected:
            raise ValueError('Validation differs from the frozen F intervals')
    output.mkdir(parents=True)
    outcomes = {}
    for arm in cfg['arms']:
        random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
        if device == 'cuda': torch.cuda.reset_peak_memory_stats()
        model = MultiInputEventTeacher(cfg['width']).to(device)
        loader = DataLoader(training, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate_events)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
        directory = output/arm; directory.mkdir()
        steps = min(8, len(loader)) if smoke else len(loader)
        epochs = 2 if smoke else cfg['epochs']
        record = dict(config=cfg, arm=arm, smoke=smoke, evaluation_partition='validation_subset_smoke' if smoke else 'validation',
            data=meta, auxiliary=aux_meta, train_ids=train_ids, evaluation_ids=val_ids,
            steps_per_epoch=steps, epochs=epochs, parameters=sum(p.numel() for p in model.parameters()),
            git_revision=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
            git_dirty=bool(subprocess.check_output(['git','status','--porcelain'], text=True).strip()),
            source_code_sha256={str(p):digest(p) for p in Path('src/acr').glob('*.py')},
            F_reference_sha256=digest(reference), device=device,
            torch=torch.__version__, numpy=np.__version__,
            protocol='Fresh shared TCN/heads, separate 12/24 projections. Replace fixed fraction of batches; no added steps. Billboard validation selects checkpoint.')
        (directory/'run.json').write_text(json.dumps(record, indent=2))
        best = -1.; history = []
        for epoch in range(1, epochs+1):
            selected = replacement_steps(steps, cfg['auxiliary_batch_fraction'] if arm == 'jaah_mixed' else 0., cfg['seed'], epoch)
            model.train(); sums = np.zeros(4); exposure = {'billboard':np.zeros(15,dtype=int), 'jaah':np.zeros(15,dtype=int)}
            started = time.perf_counter()
            for step, (x, mask, y) in enumerate(loader):
                if step >= steps: break
                source = 'billboard'
                if step in selected:
                    source = 'jaah'
                    # Aux sampling cannot alter the Billboard sampler's RNG sequence.
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(cfg['seed']+epoch*10000+step)
                        indices = torch.randint(len(auxiliary), (len(y),)).tolist()
                        x, mask, y = collate_events([auxiliary[i] for i in indices])
                buckets = torch.where(y[:,0] == 0, 14, (y[:,0]-1)%14)
                exposure[source] += torch.bincount(buckets, minlength=15).numpy()
                terms = losses(model(x.to(device), mask.to(device)), y.to(device))
                loss = sum(terms)
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
                sums += [t.item() for t in terms]
            metrics, _ = evaluate(model, validation, cfg['batch_size'], device)
            row = dict(epoch=epoch, seconds=time.perf_counter()-started, head_losses=(sums/steps).tolist(),
                auxiliary_steps=len(selected), sampled_quality_events={s:dict(zip([*QUALITIES,'N'],map(int,v))) for s,v in exposure.items()}, **metrics)
            history.append(row); print(json.dumps(dict(arm=arm, **row)), flush=True)
            (directory/'history.json').write_text(json.dumps(history, indent=2))
            if metrics['event_accuracy'] > best:
                best = metrics['event_accuracy']
                torch.save(dict(state_dict=model.state_dict(), epoch=epoch, width=cfg['width'], classes=CLASSES,
                    model='multi_input_event_TCN'), directory/'best.pt')
        checkpoint = torch.load(directory/'best.pt', weights_only=True, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        metrics, rows = evaluate(model, validation, cfg['batch_size'], device, directory)
        metrics.update(best_epoch=checkpoint['epoch'], smoke=smoke, evaluation_partition=record['evaluation_partition'],
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)
        if not smoke: compare(reference, rows, directory)
        (directory/'metrics.json').write_text(json.dumps(metrics, indent=2))
        outcomes[arm] = metrics
    summary = dict(seed=cfg['seed'], smoke=smoke, results=outcomes,
        deltas={k:outcomes['jaah_mixed'][k]-outcomes['billboard_control'][k] for k in ('event_accuracy','rare_macro_event_recall','macro_quality_event_recall')
                if outcomes['jaah_mixed'][k] is not None and outcomes['billboard_control'][k] is not None},
        note='Smoke proves execution only.' if smoke else 'Single-seed screening; inspect per-quality precision and F/E complementarity before seed repeats.')
    (output/'comparison.json').write_text(json.dumps(summary, indent=2)); print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('configs/p2-jaah-pilot.json'))
    p.add_argument('--cache', type=Path, default=Path('data/processed/p1'))
    p.add_argument('--aux-cache', type=Path, default=Path('data/processed/p2-jaah'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); run(a.config, a.cache, a.aux_cache, a.output, a.device, a.smoke)
