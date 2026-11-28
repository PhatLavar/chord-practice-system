"""Train independently initialized oracle-event teacher E."""

import argparse
import csv
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from acr.data.billboard.corpus import table
from acr.data.events import partitions, load_events, Events, collate_events
from acr.evaluation.frame import component_scores, probabilities
from acr.vocabulary import CLASSES, QUALITIES
from acr.models.tcn import EventTeacher
from acr.models.components import losses, decode, targets

FIELDS = "id start end reference prediction gate root triad extension".split()


@torch.no_grad()
def evaluate(model, dataset, batch_size, device, output=None):
    model.eval()
    parts = [[] for _ in range(4)]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_events)
    for x, mask, _ in loader:
        for collected, head in zip(parts, model(x.to(device), mask.to(device))):
            collected.append(head[:, :, 0].cpu())

    logits = [torch.cat(p).numpy().T for p in parts]
    probs = [probabilities(p) for p in logits]
    predicted = decode([torch.from_numpy(p).unsqueeze(0) for p in logits])[0].numpy()
    y = np.array([e[-1] for e in dataset.events])
    rows = []
    for i, event in enumerate(dataset.events):
        rows.append(
            [
                event[0],
                event[1],
                event[2],
                int(y[i]),
                int(predicted[i]),
                *[int(p[:, i].argmax()) for p in probs],
            ]
        )

    metrics = dict(event_accuracy=float((predicted == y).mean()), scored_events=len(y))
    with Path("reports/p0/qualities.csv").open() as stream:
        rare = {r["quality"] for r in csv.DictReader(stream) if r["rare"] == "True"}
    quality_rows = []
    recalls = {}
    components = component_scores(probs, y)
    for j, q in enumerate(QUALITIES):
        mask = (y > 0) & ((y - 1) % len(QUALITIES) == j)
        predmask = (predicted > 0) & ((predicted - 1) % len(QUALITIES) == j)
        n = int(mask.sum())
        correct = int((mask & (predicted == y)).sum())
        recalls[q] = correct / n if n else None
        songs = len({e[0] for e, m in zip(dataset.events, mask) if m})
        quality_rows.append(
            [
                q,
                n,
                songs,
                correct,
                correct / n if n else "",
                correct / int(predmask.sum()) if predmask.any() else "",
                *[float(components[mask, k].mean()) if n else "" for k in range(3)],
            ]
        )

    rare_recalls = [recalls[q] for q in rare if recalls[q] is not None]
    metrics["rare_macro_event_recall"] = float(np.mean(rare_recalls)) if rare_recalls else None
    metrics["macro_quality_event_recall"] = float(
        np.mean([v for v in recalls.values() if v is not None])
    )

    if output:
        table(output / "events.csv", " ".join(FIELDS), rows)
        table(
            output / "per_quality.csv",
            "quality events songs correct_complete_events recall precision root_accuracy triad_accuracy extension_accuracy",
            quality_rows,
        )
        np.savez_compressed(
            output / "event_logits.npz",
            reference=y,
            prediction=predicted,
            **{f"logits_{i}": p for i, p in enumerate(logits)},
        )

    return metrics, rows


def compare(f_reference, e_rows, output):
    with f_reference.open() as f:
        f_rows = list(csv.DictReader(f))
    f_map = {(r["id"], float(r["start"]), float(r["end"])): r for r in f_rows}
    if len(f_map) != len(f_rows) or len(f_rows) != len(e_rows):
        raise ValueError("F/E event counts or identities differ")
    counts = {}
    for values in e_rows:
        e = dict(zip(FIELDS, values))
        key = (e["id"], e["start"], e["end"])
        f = f_map.pop(key)
        ref = int(e["reference"])
        if int(f["reference"]) != ref:
            raise ValueError("F/E reference mismatch")
        q = "N" if ref == 0 else QUALITIES[(ref - 1) % len(QUALITIES)]
        # The existing component target mapper is shared with F.
        target = [int(t.item()) for t in targets(torch.tensor(ref))]
        for metric, label in [
            ("complete", ref),
            ("gate", target[0]),
            ("root", target[1]),
            ("triad", target[2]),
            ("extension", target[3]),
        ]:
            if label < 0:
                continue
            field = "prediction" if metric == "complete" else metric
            a, b = int(f[field]) == label, int(e[field]) == label
            bucket = counts.setdefault((q, metric), [0, 0, 0, 0])
            bucket[0 if a and b else 1 if a else 2 if b else 3] += 1
    if f_map:
        raise ValueError("Unmatched F events")
    table(
        output / "complementarity.csv",
        "quality metric both_correct F_only E_only both_wrong",
        [[q, m, *v] for (q, m), v in sorted(counts.items())],
    )


def train(cache, config, output, device):
    cfg = json.loads(config.read_text())
    meta = partitions(cache)

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(4)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Enable a GPU")

    train_ids = sorted(i for i, p in meta["recordings"].items() if p == "train_labeled")
    eval_ids = sorted(i for i, p in meta["recordings"].items() if p == "validation")
    if cfg["mode"] == "overfit":
        train_ids = train_ids[: cfg["songs"]]
        eval_ids = train_ids
    elif cfg["mode"] != "oracle":
        raise ValueError("Unknown mode")
    if cfg["sampling"] not in ("natural", "quality_song_mixture"):
        raise ValueError("Unknown event sampling")

    songs, events, empty = load_events(
        cache, Path("data/raw/full.tar.xz"), sorted(set(train_ids + eval_ids))
    )
    train_events = [e for e in events if e[0] in train_ids]
    eval_events = [e for e in events if e[0] in eval_ids]
    natural = Events(songs, train_events)
    probability = (
        cfg.get("balanced_probability", 0.5) if cfg["sampling"] == "quality_song_mixture" else 0.0
    )
    if cfg["sampling"] == "natural" and cfg.get("balanced_probability", 0) != 0:
        raise ValueError("Natural sampling cannot specify a nonzero balanced probability")
    training = Events(
        songs,
        train_events,
        balanced=cfg["sampling"] == "quality_song_mixture",
        balanced_probability=probability,
    )
    validation = Events(songs, eval_events)

    if cfg["mode"] == "oracle":
        reference = Path("reports/p2/f-reference/events.csv")
        with reference.open() as f:
            frozen = {
                (r["id"], float(r["start"]), float(r["end"]), int(r["reference"]))
                for r in csv.DictReader(f)
            }
        expected = {(e[0], e[1], e[2], e[-1]) for e in eval_events}
        if frozen != expected:
            raise ValueError("Frozen F intervals do not match E validation intervals")

    model = EventTeacher(cfg["width"]).to(device)
    loader = DataLoader(
        training, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_events
    )

    output.mkdir(parents=True, exist_ok=False)
    record = dict(
        config=cfg,
        train_ids=train_ids,
        evaluation_ids=eval_ids,
        evaluation_partition="training" if cfg["mode"] == "overfit" else "validation",
        oracle_boundaries=True,
        data=meta,
        train_events=len(train_events),
        evaluation_events=len(eval_events),
        empty_events_total=empty,
        parameters=sum(p.numel() for p in model.parameters()),
        sampling_policy=dict(
            balanced_probability=probability,
            natural_probability=1 - probability,
            event_draws_per_epoch=len(training),
            optimizer_steps_per_epoch=len(loader),
        ),
        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_dirty=bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        device=device,
        torch=torch.__version__,
        numpy=np.__version__,
        source_code_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("src/acr").rglob("*.py")
        },
    )
    (output / "run.json").write_text(json.dumps(record, indent=2))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    best = -1
    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        started = time.perf_counter()
        sums = np.zeros(4)
        exposure = np.zeros(15, dtype=np.int64)

        for x, mask, y in loader:
            observed = y[:, 0]
            buckets = torch.where(observed == 0, 14, (observed - 1) % 14)
            exposure += torch.bincount(buckets, minlength=15).numpy()

            terms = losses(model(x.to(device), mask.to(device)), y.to(device))
            loss = sum(terms)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite event loss")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            sums += [t.item() for t in terms]

        metrics, _ = evaluate(model, validation, cfg["batch_size"], device)
        row = dict(
            epoch=epoch,
            head_losses=(sums / len(loader)).tolist(),
            seconds=time.perf_counter() - started,
            sampled_quality_events=dict(zip([*QUALITIES, "N"], map(int, exposure))),
            **metrics,
        )
        history.append(row)
        print(json.dumps(row), flush=True)
        (output / "history.json").write_text(json.dumps(history, indent=2))

        if metrics["event_accuracy"] > best:
            best = metrics["event_accuracy"]
            torch.save(
                dict(
                    state_dict=model.state_dict(),
                    width=cfg["width"],
                    classes=CLASSES,
                    epoch=epoch,
                    model="oracle_event_TCN",
                ),
                output / "best.pt",
            )

    checkpoint = torch.load(output / "best.pt", weights_only=True, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    final, rows = evaluate(model, validation, cfg["batch_size"], device, output)
    train_metrics, _ = evaluate(model, natural, cfg["batch_size"], device)
    final.update(
        best_epoch=checkpoint["epoch"],
        evaluation_partition=record["evaluation_partition"],
        overfit_pass=best >= 0.95 if cfg["mode"] == "overfit" else None,
        training_event_accuracy=train_metrics["event_accuracy"],
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
    )
    if cfg["mode"] == "oracle":
        reference = Path("reports/p2/f-reference/events.csv")
        compare(reference, rows, output)
        final["F_reference_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()
    (output / "metrics.json").write_text(json.dumps(final, indent=2))
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path("data/processed/p1"))
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = p.parse_args()
    train(a.cache, a.config, a.output, a.device)
