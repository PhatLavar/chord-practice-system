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
from acr.data.events import Events, collate_events
from acr.evaluation.intervals import score
from acr.data.guitarset.labels import map_symbol
from acr.vocabulary import QUALITIES
from acr.data.guitarset.cqt import load_cache, digest
from acr.models.tcn import FrameTeacher
from acr.models.components import decode, losses
from acr.models.tcn import EventTeacher
from acr.data.windows import Windows
from acr.data.augmentation import transpose_batch
from acr.provenance import config_path

from acr.data.guitarset.cqt import event_list
from acr.evaluation.guitarset import evaluate


def resolve_cache_config(cfg):
    if "cache_config" not in cfg:
        return cfg

    cached = json.loads(config_path(cfg["cache_config"]).read_text())
    expected = dict(cached, seed=cfg["seed"])
    if {k: v for k, v in cfg.items() if k not in ("cache_config", "augmentation")} != expected:
        raise ValueError("Only seed and augmentation may differ from the cached control config")

    return cached


def train(cache, config, output, device, smoke=False):
    if output.exists():
        raise FileExistsError("Use a fresh run directory")

    cfg = json.loads(config.read_text())
    cache_cfg = resolve_cache_config(cfg)
    songs, refs, meta = load_cache(cache, cache_cfg, smoke)

    augmentation = cfg.get("augmentation")
    if augmentation:
        if (
            augmentation != {"method": "cqt_shift", "semitones": list(range(-5, 7))}
            or cfg["bins_per_octave"] % 12
        ):
            raise ValueError("Unknown augmentation protocol")
        if any(np.any(np.all(s["x"] == 0, axis=1)) for s in songs.values()):
            raise ValueError("All-zero real frame would be ambiguous with normalized padding")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Enable GPU")

    torch.set_num_threads(4)
    train_songs = {i: s for i, s in songs.items() if meta["recordings"][i] == "train"}
    validation = {i: s for i, s in songs.items() if meta["recordings"][i] == "validation"}

    output.mkdir(parents=True)
    summary = {}
    (output / "validation_references.json").write_text(
        json.dumps({i: refs[i] for i in validation}, indent=2) + "\n"
    )

    for teacher, cls in [("F", FrameTeacher), ("E", EventTeacher)]:
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        model = cls(width=cfg["width"], input_channels=cfg["n_bins"]).to(device)
        augmentation_rng = torch.Generator().manual_seed(cfg["seed"] + 1)

        if teacher == "F":
            dataset = Windows(train_songs, cfg["window"], model.context)
            collate = None
        else:
            supported = [e for e in event_list(train_songs, refs) if e[-1] >= 0]
            dataset = Events(train_songs, supported, context=model.context)
            collate = collate_events
        loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate)
        if not len(loader):
            raise ValueError("No training batches")

        directory = output / teacher
        directory.mkdir()
        epochs = 2 if smoke else cfg["epochs"]
        steps = min(2, len(loader)) if smoke else len(loader)
        record = dict(
            config=cfg,
            teacher=teacher,
            smoke=smoke,
            epochs=epochs,
            steps_per_epoch=steps,
            train_ids=sorted(train_songs),
            validation_ids=sorted(validation),
            cache_metadata=meta,
            torch=torch.__version__,
            device=device,
            git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            source_sha256={p.as_posix(): digest(p) for p in Path("src/acr").rglob("*.py")},
            augmentation=augmentation,
            protocol="Fresh CQT-TCN; natural sampling; complete labels only; select validation duration-weighted Tetrads; no outer test. Optional CQT shift only during optimizer updates; clean train/validation diagnostics.",
        )
        (directory / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        best = -1
        history = []

        for epoch in range(1, epochs + 1):
            model.train()
            sums = np.zeros(4)
            shift_counts = {str(s): 0 for s in augmentation["semitones"]} if augmentation else None

            for step, batch in enumerate(loader):
                if step >= steps:
                    break

                if teacher == "F":
                    x, y = batch
                else:
                    x, mask, y = batch

                if augmentation:
                    choices = augmentation["semitones"]
                    shifts = [
                        choices[i]
                        for i in torch.randint(
                            len(choices), (len(x),), generator=augmentation_rng
                        ).tolist()
                    ]
                    for shift in shifts:
                        shift_counts[str(shift)] += 1
                    stats = meta["normalization"]
                    floor = (np.log(1e-6) - stats["mean"]) / stats["std"]
                    x, y = transpose_batch(x, y, shifts, cfg["bins_per_octave"] // 12, float(floor))

                if teacher == "F":
                    heads = [
                        h[:, :, model.context : model.context + cfg["window"]]
                        for h in model(x.to(device))
                    ]
                else:
                    heads = model(x.to(device), mask.to(device))

                terms = losses(heads, y.to(device))
                loss = sum(terms)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss")

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                sums += [float(t.detach()) for t in terms]

            metrics = evaluate(model, teacher, validation, refs, device, cfg["batch_size"])
            value = metrics["metrics"]["tetrads"]["score"]
            if value is None:
                raise ValueError("No eligible validation duration")

            row = dict(epoch=epoch, head_losses=(sums / steps).tolist(), validation_tetrads=value)
            if augmentation:
                row["augmentation_shift_counts"] = shift_counts
            if cfg.get("track_training_metrics", False):
                fitted = evaluate(model, teacher, train_songs, refs, device, cfg["batch_size"])
                row["training_metrics"] = fitted
                row["validation_metrics"] = metrics

            history.append(row)
            progress = dict(
                teacher=teacher,
                epoch=epoch,
                head_losses=row["head_losses"],
                validation_tetrads=value,
            )
            if "training_metrics" in row:
                progress["training_tetrads"] = row["training_metrics"]["metrics"]["tetrads"][
                    "score"
                ]

            print(json.dumps(progress), flush=True)
            (directory / "history.json").write_text(json.dumps(history, indent=2) + "\n")

            if value > best:
                best = value
                torch.save(
                    dict(state_dict=model.state_dict(), epoch=epoch, config=cfg, teacher=teacher),
                    directory / "best.pt",
                )

        checkpoint = torch.load(directory / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        predictions = directory / "predictions"
        predictions.mkdir()
        metrics = evaluate(model, teacher, validation, refs, device, cfg["batch_size"], predictions)
        metrics.update(
            best_epoch=checkpoint["epoch"],
            evaluation_partition="validation_subset_smoke" if smoke else "validation",
            smoke=smoke,
        )
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        summary[teacher] = metrics

    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/p2/p2-guitarset-cqt.json"))
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    train(a.cache, a.config, a.output, a.device, a.smoke)
