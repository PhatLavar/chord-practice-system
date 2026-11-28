"""Training-only CQT memorization diagnostic; not a generalization benchmark."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from acr.data.events import Events, collate_events
from acr.vocabulary import QUALITIES
from acr.data.guitarset.cqt import load_cache, digest
from acr.models.tcn import FrameTeacher
from acr.models.components import decode, losses
from acr.models.tcn import EventTeacher
from acr.data.guitarset.cqt import event_list


def select_events(events):
    """Two short events per available quality, preferring different roots/recordings."""
    selected = []
    for q in range(len(QUALITIES)):
        candidates = sorted(
            (e for e in events if e[-1] > 0 and (e[-1] - 1) % 14 == q and 4 <= e[4] - e[3] <= 108),
            key=lambda e: (e[4] - e[3], e[0], e[1]),
        )
        if not candidates:
            continue
        first = candidates[0]
        selected.append(first)
        rest = [e for e in candidates if e != first]
        if rest:
            selected.append(
                min(
                    rest,
                    key=lambda e: (
                        (e[-1] - 1) // 14 == (first[-1] - 1) // 14,
                        e[0] == first[0],
                        e[4] - e[3],
                        e[0],
                        e[1],
                    ),
                )
            )
    if not selected:
        raise ValueError("No eligible training events")
    return selected


def run(cache, output, steps=500):
    if output.exists():
        raise FileExistsError("Use a fresh diagnostic output")
    cfg = json.loads(Path("configs/p2/p2-guitarset-cqt.json").read_text())
    songs, refs, meta = load_cache(cache, cfg, False)
    songs = {rid: s for rid, s in songs.items() if meta["recordings"][rid] == "train"}
    events = select_events(event_list(songs, refs))
    assert all(meta["recordings"][e[0]] == "train" for e in events)
    assert all(np.all(songs[e[0]]["y"][e[3] : e[4]] == e[-1]) for e in events)
    x, mask, y = collate_events([Events(songs, events)[i] for i in range(len(events))])
    torch.set_num_threads(4)
    output.mkdir(parents=True)
    report = dict(
        protocol="Fixed training-only gold-event crops; fresh models; same full batch each update. F supervises frames inside event, E pools event features. No validation/test selection.",
        seed=cfg["seed"],
        max_updates=steps,
        learning_rate=cfg["learning_rate"],
        pass_rule="F >= 95% selected-frame accuracy and every selected event pooled correctly; E every selected event correct.",
        cache_metadata_sha256=digest(cache / "metadata.json"),
        source_sha256={p.as_posix(): digest(p) for p in Path("src/acr").rglob("*.py")},
        events=[
            dict(
                recording=e[0],
                start=e[1],
                end=e[2],
                frames=e[4] - e[3],
                label=e[-1],
                quality=QUALITIES[(e[-1] - 1) % 14],
            )
            for e in events
        ],
        results={},
    )
    for teacher, cls in [("F", FrameTeacher), ("E", EventTeacher)]:
        torch.manual_seed(cfg["seed"])
        model = cls(width=cfg["width"], input_channels=cfg["n_bins"])
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        target = y.expand_as(mask).masked_fill(~mask, -100) if teacher == "F" else y
        history = []
        for step in range(1, steps + 1):
            model.train()
            heads = model(x) if teacher == "F" else model(x, mask)
            terms = losses(heads, target)
            loss = sum(terms)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            if step == 1 or step % 10 == 0 or step == steps:
                model.eval()
                with torch.no_grad():
                    heads = model(x) if teacher == "F" else model(x, mask)
                    pred = decode(heads)
                    if teacher == "F":
                        accuracy = float((pred[mask] == target[mask]).float().mean())
                        pooled = [
                            ((h.softmax(1) * mask[:, None, :]).sum(2) / mask.sum(1)[:, None])
                            .clamp_min(1e-30)
                            .log()[:, :, None]
                            for h in heads
                        ]
                        event_pred = decode(pooled)
                    else:
                        accuracy = float((pred == y).float().mean())
                        event_pred = pred
                    correct = (event_pred == y).flatten()
                passed = bool(correct.all()) and accuracy >= 0.95
                row = dict(
                    update=step,
                    loss=float(loss.detach()),
                    accuracy=accuracy,
                    event_accuracy=float(correct.float().mean()),
                    passed=passed,
                )
                history.append(row)
                if step == 1 or step % 50 == 0 or passed:
                    print(json.dumps(dict(teacher=teacher, **row)), flush=True)
                if passed:
                    break
        torch.save(model.state_dict(), output / (teacher + ".pt"))
        report["results"][teacher] = dict(
            final=history[-1], history=history, event_predictions=event_pred.flatten().tolist()
        )
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path("data/processed/guitarset-cqt-fold0"))
    p.add_argument("--output", type=Path, default=Path("runs/p2-guitarset-cqt-memorization"))
    p.add_argument("--steps", type=int, default=500)
    a = p.parse_args()
    if a.steps < 1:
        p.error("--steps must be positive")
    run(a.cache, a.output, a.steps)
