"""Evaluate frozen player-held-out checkpoints on train; no optimizer or selection."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from acr.evaluation.guitarset import run as diagnose
from acr.data.guitarset.cqt import load_cache, digest
from acr.models.tcn import FrameTeacher
from acr.models.tcn import EventTeacher
from acr.evaluation.guitarset import evaluate
from acr.provenance import config_path


def run(cache, source, output):
    if output.exists():
        raise FileExistsError("Use a fresh evaluation directory")
    cfg = json.loads((source / "F/run.json").read_text())["config"]
    cache_cfg = (
        json.loads(config_path(cfg["cache_config"]).read_text()) if "cache_config" in cfg else cfg
    )
    songs, refs, meta = load_cache(cache, cache_cfg, False)
    training = {rid: s for rid, s in songs.items() if meta["recordings"][rid] == "train"}
    if not training:
        raise ValueError("Empty training partition")
    output.mkdir(parents=True)
    references = output / "training_references.json"
    references.write_text(json.dumps({rid: refs[rid] for rid in training}, indent=2) + "\n")
    torch.set_num_threads(4)
    provenance = dict(
        protocol="Frozen validation-selected checkpoints evaluated on training partition; no updates.",
        cache_metadata_sha256=digest(cache / "metadata.json"),
        training_ids=sorted(training),
        platform_note="Local CPU CQT may differ slightly numerically from Kaggle; verify source hashes and compare saved epoch training scores.",
        source_sha256={p.as_posix(): digest(p) for p in Path("src/acr").rglob("*.py")},
        jobs={},
    )
    for teacher, cls in [("F", FrameTeacher), ("E", EventTeacher)]:
        record = json.loads((source / teacher / "run.json").read_text())
        original = record["cache_metadata"]
        if record["config"] != cfg or record["train_ids"] != sorted(training):
            raise ValueError("Wrong checkpoint protocol")
        for key in ["source_sha256", "audio_sha256", "recordings"]:
            if original[key] != meta[key]:
                raise ValueError("Local cache differs from imported source: " + key)
        manifest = Path(cfg["folds_path"])
        committed = subprocess.check_output(
            ["git", "show", record["git_revision"] + ":" + manifest.as_posix()]
        )
        if hashlib.sha256(committed).hexdigest() != original[
            "folds_sha256"
        ] or manifest.read_bytes().replace(b"\r\n", b"\n") != committed.replace(b"\r\n", b"\n"):
            raise ValueError("Manifest differs beyond platform line endings")
        if not np.allclose(
            [original["normalization"][k] for k in ["mean", "std"]],
            [meta["normalization"][k] for k in ["mean", "std"]],
            atol=1e-5,
            rtol=0,
        ):
            raise ValueError("Unexpected normalization difference")
        checkpoint = torch.load(source / teacher / "best.pt", map_location="cpu", weights_only=True)
        saved = json.loads((source / teacher / "metrics.json").read_text())
        if checkpoint["epoch"] != saved["best_epoch"]:
            raise ValueError("Wrong selected checkpoint")
        model = cls(width=cfg["width"], input_channels=cfg["n_bins"])
        model.load_state_dict(checkpoint["state_dict"])
        directory = output / teacher
        directory.mkdir()
        predictions = directory / "predictions"
        predictions.mkdir()
        metrics = evaluate(model, teacher, training, refs, "cpu", cfg["batch_size"], predictions)
        metrics.update(evaluation_partition="training", best_epoch=checkpoint["epoch"])
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        history = json.loads((source / teacher / "history.json").read_text())
        expected = history[checkpoint["epoch"] - 1]["training_metrics"]["metrics"]["tetrads"][
            "score"
        ]
        actual = metrics["metrics"]["tetrads"]["score"]
        provenance["jobs"][teacher] = dict(
            checkpoint_sha256=digest(source / teacher / "best.pt"),
            selected_epoch=checkpoint["epoch"],
            original_training_tetrads=expected,
            local_training_tetrads=actual,
            difference=actual - expected,
        )
        print(teacher, provenance["jobs"][teacher], flush=True)
    diagnose(output, output / "diagnostics", references)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path("data/processed/guitarset-cqt-player"))
    p.add_argument("--source", type=Path, default=Path("runs/p2-guitarset-cqt-player"))
    p.add_argument("--output", type=Path, default=Path("runs/p2-guitarset-player-training-fit"))
    a = p.parse_args()
    run(a.cache, a.source, a.output)
