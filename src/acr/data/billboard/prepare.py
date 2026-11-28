"""Prepare P1 arrays from the frozen P0 inputs. No test or hidden labels are read."""

import argparse
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from acr.data.billboard.audit import aligned_frames, read_events
from acr.vocabulary import COMPONENTS, mapping
from acr.vocabulary import CLASSES, QUALITIES, TRIADS, EXTENSIONS, encode


def prepare(raw, manifest, hashes, output):
    expected = json.loads(hashes.read_text())["source_sha256"]
    for name, digest in expected.items():
        with (raw / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(f"Source checksum mismatch: {name}")
    with manifest.open() as stream:
        rows = list(csv.DictReader(stream))
    selected = {
        r["id"]: r["partition"]
        for r in rows
        if r["id"] == r["representative"] and r["partition"] in ("train_labeled", "validation")
    }
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        source_sha256=expected,
        classes=CLASSES,
        recordings=selected,
    )
    with (
        tarfile.open(raw / "full.tar.xz") as lab,
        tarfile.open(raw / "billboard-2.0-chordino.tar.xz", "r|xz") as features,
    ):
        labels = {Path(m.name).parent.name: m for m in lab if m.name.endswith("/full.lab")}
        found = set()
        for member in features:
            rid = Path(member.name).parent.name
            if not member.name.endswith("/bothchroma.csv") or rid not in selected:
                continue
            events = read_events(lab.extractfile(labels[rid]).read().decode().splitlines())
            frames = list(
                aligned_frames(io.StringIO(features.extractfile(member).read().decode()), events)
            )
            times = np.array([f[0] for f in frames], dtype=np.float64)
            x = np.array([f[2:] for f in frames], dtype=np.float32)
            # Normalize each 12-bin block separately, without corpus-wide fitted statistics.
            x = x.reshape(-1, 2, 12)
            x /= np.maximum(x.sum(axis=-1, keepdims=True), 1e-8)
            y = np.array([encode(f[1]) for f in frames], dtype=np.int64)
            np.savez_compressed(output / f"{rid}.npz", x=x.reshape(-1, 24), y=y, times=times)
            found.add(rid)
    if found != selected.keys():
        raise ValueError(f"Missing features: {selected.keys() - found}")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Prepared {len(found)} recordings in {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--output", type=Path, default=Path("data/processed/p1"))
    a = p.parse_args()
    prepare(a.raw, Path("splits/billboard-v1.csv"), Path("reports/p0/summary.json"), a.output)
