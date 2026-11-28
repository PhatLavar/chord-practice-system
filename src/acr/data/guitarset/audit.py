"""Inventory pinned GuitarSet archives without changing training data or splits."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import wave
import zipfile
from urllib.request import urlopen

import numpy as np
from acr.vocabulary import mapping
from acr.data.billboard.corpus import table

MD5 = {
    "annotation.zip": "b39b78e63d3446f2e54ddb7a54df9b10",
    "audio_mono-mic.zip": "275966d6610ac34999b58426beb119c3",
}
ERRATA = {"04_BN3-154-E_comp", "04_Jazz1-200-B_comp"}


def download(raw):
    raw.mkdir(parents=True, exist_ok=True)
    for name in MD5:
        target = raw / name
        if target.exists():
            continue
        temporary = raw / (name + ".part")
        with (
            urlopen(
                f"https://zenodo.org/records/3371780/files/{name}?download=1", timeout=60
            ) as response,
            temporary.open("wb") as stream,
        ):
            while block := response.read(4 * 1024 * 1024):
                stream.write(block)
        with temporary.open("rb") as stream:
            if hashlib.file_digest(stream, "md5").hexdigest() != MD5[name]:
                raise ValueError(f"Download checksum mismatch: {name}")
        temporary.replace(target)


def audit(raw, output):
    if output.exists():
        raise FileExistsError("Use a fresh report directory")
    hashes = {}
    for name, expected in MD5.items():
        with (raw / name).open("rb") as f:
            if hashlib.file_digest(f, "md5").hexdigest() != expected:
                raise ValueError(f"Checksum mismatch: {name}")
        with (raw / name).open("rb") as f:
            hashes[name] = hashlib.file_digest(f, "sha256").hexdigest()
    inventory = []
    symbols = Counter()
    seconds = Counter()
    qualities = Counter()
    support = defaultdict(set)
    with (
        zipfile.ZipFile(raw / "annotation.zip") as labels,
        zipfile.ZipFile(raw / "audio_mono-mic.zip") as audio,
    ):
        names = sorted(
            n for n in labels.namelist() if n.endswith(".jams") and not n.startswith("__MACOSX")
        )
        wavs = {
            Path(n).stem.removesuffix("_mic"): n
            for n in audio.namelist()
            if n.endswith(".wav") and not n.startswith("__MACOSX")
        }
        if len(names) != 360 or {Path(n).stem for n in names} != set(wavs):
            raise ValueError("Unexpected recording inventory/audio pairing")
        for name in names:
            rid = Path(name).stem
            doc = json.loads(labels.read(name))
            with audio.open(wavs[rid]) as f, wave.open(f) as w:
                duration = w.getnframes() / w.getframerate()
                rate, channels = w.getframerate(), w.getnchannels()
            annotations = [a for a in doc["annotations"] if a["namespace"] == "chord"]
            if len(annotations) != 2:
                raise ValueError(f"Expected two chord layers: {rid}")
            if (
                sum(
                    "Chord sheet-informed" in a["annotation_metadata"]["annotation_rules"]
                    for a in annotations
                )
                != 1
            ):
                raise ValueError(f"Ambiguous chord layer metadata: {rid}")
            status = "excluded_timing_erratum" if rid in ERRATA else "candidate_audio_verified"
            ends = []
            for a in annotations:
                rules = a["annotation_metadata"]["annotation_rules"]
                layer = "performed" if "Chord sheet-informed" in rules else "instructed"
                previous = 0.0
                for e in a["data"]:
                    start, length = float(e["time"]), float(e["duration"])
                    end = start + length
                    if (
                        not np.isfinite([start, length]).all()
                        or start < -1e-8
                        or length <= 0
                        or start < previous - 1e-6
                    ):
                        raise ValueError(f"Invalid interval: {rid}")
                    if end > duration + 0.1:
                        raise ValueError(f"Labels exceed audio: {rid}")
                    previous = end
                    symbol = e["value"]
                    quality = mapping(symbol)[0]
                    key = (status, layer, symbol, quality)
                    symbols[key] += 1
                    seconds[key] += length
                    qualities[(status, layer, quality)] += length
                    support[key].add(rid)
                ends.append(previous)
            player, sheet, mode = rid.split("_")
            # Conservative candidate key: same style/progression across key/tempo/player.
            # Cross-style versions of the three underlying progressions still need grouping.
            inventory.append(
                [
                    rid,
                    player,
                    sheet.split("-")[0],
                    mode,
                    status,
                    rate,
                    channels,
                    duration,
                    doc["file_metadata"]["duration"],
                    max(ends),
                    wavs[rid],
                ]
            )
    output.mkdir(parents=True)
    table(
        output / "recordings.csv",
        "id player candidate_group mode status sample_rate channels audio_seconds metadata_seconds annotation_end audio_member",
        inventory,
    )
    table(
        output / "symbols.csv",
        "status layer symbol mapped_quality events seconds recordings",
        [list(k) + [v, seconds[k], len(support[k])] for k, v in sorted(symbols.items())],
    )
    table(
        output / "qualities.csv",
        "status layer quality seconds",
        [list(k) + [v] for k, v in sorted(qualities.items())],
    )
    summary = dict(
        release="https://zenodo.org/records/3371780",
        source_sha256=hashes,
        recordings=len(inventory),
        candidates=sum(r[4] == "candidate_audio_verified" for r in inventory),
        excluded=sorted(ERRATA),
        audio_seconds=sum(r[7] for r in inventory),
        license="CC-BY-4.0 (Zenodo release metadata)",
        status="Inventory only; no features, training admission or folds created",
        grouping="Candidate style/progression groups only; cross-style progression relationships require review before folds",
        annotation="Performed labels are sheet-informed, semi-automatic; preserve instructed separately",
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("data/raw/guitarset"))
    p.add_argument("--output", type=Path, default=Path("reports/p2/guitarset-inventory"))
    p.add_argument("--download", action="store_true")
    a = p.parse_args()
    if a.download:
        download(a.raw)
    audit(a.raw, a.output)
