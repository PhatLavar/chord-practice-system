"""Audit the pinned Billboard releases and generate the song-level P0 manifest."""

import argparse
import csv
import hashlib
import io
import itertools
import json
import random
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from audit import aligned_frames, read_events


# (triad, extension); inversions collapse to the same root and quality.
COMPONENTS = {
    "maj": ("maj", "none"), "min": ("min", "none"), "dim": ("dim", "none"),
    "aug": ("aug", "none"), "sus2": ("sus2", "none"), "sus4": ("sus4", "none"),
    "maj6": ("maj", "6"), "min6": ("min", "6"), "7": ("maj", "b7"),
    "min7": ("min", "b7"), "maj7": ("maj", "7"), "minmaj7": ("min", "7"),
    "hdim7": ("dim", "b7"), "dim7": ("dim", "bb7"),
}
PITCH = dict(C=0, D=2, E=4, F=5, G=7, A=9, B=11)


def mapping(symbol):
    if symbol in {"N", "X"}:
        return symbol, "", "", ""
    root, _, quality = symbol.partition("/")[0].partition(":")
    if quality not in COMPONENTS:
        return "OOV", "", "", ""
    pitch = (PITCH[root[0]] + root.count("#") - root.count("b")) % 12
    return quality, pitch, *COMPONENTS[quality]


def table(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header.split())
        w.writerows(rows)


def run(raw, out):
    out.mkdir(parents=True, exist_ok=True)
    with (raw / "index.csv").open(encoding="utf-8-sig") as f:
        index = {r["id"].zfill(4): r for r in csv.DictReader(f)}
    events = {}
    inventory = []
    valid = {}
    parent = {i: i for i in index}

    def group(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    with tarfile.open(raw / "full.tar.xz") as lab, zipfile.ZipFile(raw / "kaggle-v1.zip") as z:
        labels = {Path(m.name).parent.name: m for m in lab if m.name.endswith("/full.lab")}
        chroma = {Path(n).parent.name: n for n in z.namelist() if n.endswith("/bothchroma.csv")}
        assert (labels.keys() | chroma.keys()) <= index.keys()
        for i, meta in sorted(index.items()):
            frames = covered = zero = 0
            end_delta = ""
            status = "eligible"
            if i not in labels or i not in chroma:
                status = "missing " + "/".join(k for k, source in (("labels", labels), ("features", chroma)) if i not in source)
            else:
                data = z.read(chroma[i])
                song = tuple(meta[k].strip().casefold() for k in ("artist", "title"))
                if not all(song):
                    raise ValueError(f"Missing song identity for {i}")
                for key in (song, hashlib.sha256(data).digest()):
                    other = seen.setdefault(key, i)
                    a, b = sorted([group(i), group(other)])
                    parent[b] = a
                try:
                    lines = lab.extractfile(labels[i]).read().decode().splitlines()
                    zero = sum(float(a) == float(b) for a,b,_ in (line.split() for line in lines if line.strip()))
                    ev = read_events(lines)
                    last = 0
                    for time, label, *_ in aligned_frames(io.StringIO(data.decode()), ev):
                        frames += 1
                        covered += bool(label)
                        last = time
                    end_delta = round(last - ev[-1][1], 6)
                    if not frames or covered / frames < 0.99 or abs(end_delta) > 1:
                        status = "alignment review"
                    else:
                        valid[i] = meta
                        events[i] = ev
                except ValueError as exc:
                    status = str(exc)
            inventory.append([i, status, frames, covered, end_delta, zero])
    table(out / "inventory.csv", "id status frames labeled_frames end_delta_seconds zero_intervals", inventory)
    groups = defaultdict(list)
    for i in valid:
        groups[group(i)].append(i)
    ids = sorted(groups)
    random.Random(20261117).shuffle(ids)
    n = len(ids)
    partitions = {}
    for j, g in enumerate(ids):
        partitions[g] = ("train_labeled" if j < n//2 else "train_hidden" if j < n*7//10
                         else "validation" if j < n*85//100 else "test")
    manifest = []
    stats = defaultdict(list)
    symbols = set()
    for g, members in sorted(groups.items()):
        rep = min(members)
        split = partitions[g]
        for i in members:
            manifest.append([i, g, rep, split, valid[i]["artist"], valid[i]["title"]])
        if split != "train_labeled":
            continue
        # Merge touching repeated targets; keep different unsupported symbols distinct.
        merged = []
        for a,b,s in events[rep]:
            symbols.add(s)
            q,root,_,_ = mapping(s)
            key = (q, root, s if q == "OOV" else "")
            if merged and merged[-1][1] == a and merged[-1][2] == key:
                merged[-1][1] = b
            else:
                merged.append([a,b,key])
        for a,b,(q,root,_) in merged:
            stats[q].append((g, root, b-a))
    table(out / "manifest.csv", "id group representative partition artist title", sorted(manifest))
    total = sum(len(stats[q]) for q in COMPONENTS)
    quality_rows = []
    for q in sorted(set(stats) | COMPONENTS.keys()):
        v = stats[q]
        quality_rows.append([q,len(v),len({g for g,_,_ in v}),len({r for _,r,_ in v if r != ""}),
                             round(sum(d for _,_,d in v),3), q in COMPONENTS and 0 < len(v) < total*.01])
    table(out / "qualities.csv", "quality events songs roots seconds rare", quality_rows)
    table(out / "symbols.csv", "symbol quality root_pc triad extension", ([s,*mapping(s)] for s in sorted(symbols)))
    artists = defaultdict(set)
    for _,_,_,split,artist,_ in manifest:
        artists[split].add(artist.casefold())
    hashes = {}
    for name in ("index.csv", "full.tar.xz", "kaggle-v1.zip"):
        with (raw / name).open("rb") as input_file:
            hashes[name] = hashlib.file_digest(input_file, "sha256").hexdigest()
    summary = dict(groups=dict(Counter(partitions.values())), eligible_slots=len(valid),
                   paired=len(labels.keys() & chroma.keys()), indexed=len(index),
                   artist_credit_overlap={f"{a}/{b}":len(artists[a]&artists[b]) for a,b in itertools.combinations(sorted(artists),2)},
                   source_sha256=hashes)
    (out / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--output", type=Path, default=Path("runs/p0-corpus"))
    args = p.parse_args()
    run(args.raw, args.output)
