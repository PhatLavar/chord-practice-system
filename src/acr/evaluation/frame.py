"""Diagnose saved teacher-F validation logits on frames and reference intervals."""

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
from acr.data.billboard.audit import read_events
from acr.vocabulary import COMPONENTS
from acr.data.billboard.corpus import table
from acr.vocabulary import CLASSES, QUALITIES, TRIADS, EXTENSIONS, encode


def probabilities(logits):
    shifted = logits - logits.max(axis=0, keepdims=True)
    values = np.exp(shifted.astype(np.float64))
    return values / values.sum(axis=0, keepdims=True)


def decode_components(probs):
    gate, root, triad, extension = probs
    quality_scores = np.stack(
        [
            np.log(triad[TRIADS.index(COMPONENTS[q][0])].clip(1e-300))
            + np.log(extension[EXTENSIONS.index(COMPONENTS[q][1])].clip(1e-300))
            for q in QUALITIES
        ]
    )
    chord = 1 + root.argmax(axis=0) * len(QUALITIES) + quality_scores.argmax(axis=0)
    return np.where(gate[1] > gate[0], 0, chord)


def component_scores(probs, reference):
    # Conditional component correctness ignores the no-chord gate.
    ids = np.asarray(reference)
    quality = (ids - 1) % len(QUALITIES)
    triad = np.array([TRIADS.index(COMPONENTS[q][0]) for q in QUALITIES])
    ext = np.array([EXTENSIONS.index(COMPONENTS[q][1]) for q in QUALITIES])
    return np.stack(
        [
            probs[1].argmax(axis=0) == (ids - 1) // len(QUALITIES),
            probs[2].argmax(axis=0) == triad[quality],
            probs[3].argmax(axis=0) == ext[quality],
        ],
        axis=-1,
    )


def diagnose(run, labels, output):
    metadata = json.loads((run / "run.json").read_text())
    if metadata["evaluation_partition"] != "validation":
        raise ValueError("This diagnostic requires the recorded validation run.")
    if metadata["data"]["classes"] != CLASSES:
        raise ValueError("Vocabulary mismatch")
    with labels.open("rb") as f:
        if (
            hashlib.file_digest(f, "sha256").hexdigest()
            != metadata["data"]["source_sha256"]["full.tar.xz"]
        ):
            raise ValueError("Annotation checksum mismatch")
    output.mkdir(parents=True, exist_ok=False)
    frames = {q: np.zeros(5, dtype=np.int64) for q in QUALITIES}
    events = {q: np.zeros(5, dtype=np.int64) for q in QUALITIES}
    songs = {q: set() for q in QUALITIES}
    event_rows = []
    reference_total = skipped = 0
    near = np.zeros(2, dtype=np.int64)
    interior = np.zeros(2, dtype=np.int64)
    gate_correct = gate_total = 0
    frame_confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    with tarfile.open(labels) as archive:
        members = {Path(m.name).parent.name: m for m in archive if m.name.endswith("/full.lab")}
        for rid in sorted(metadata["evaluation_ids"]):
            with np.load(run / "predictions" / f"{rid}.npz") as data:
                times, y, prediction = data["times"], data["reference"], data["prediction"]
                probs = [probabilities(data[f"logits_{i}"]) for i in range(4)]
            assert np.array_equal(decode_components(probs), prediction)
            ev = reference_events(archive.extractfile(members[rid]).read().decode().splitlines())
            rebuilt = np.full(len(times), -100)
            for start, end, label in ev:
                rebuilt[(times >= start) & (times < end)] = label
            assert np.array_equal(rebuilt, y), f"Saved reference differs for {rid}"
            valid = y >= 0
            np.add.at(frame_confusion, (y[valid], prediction[valid]), 1)
            gate_correct += int(((probs[0].argmax(0) == (y == 0)) & valid).sum())
            gate_total += int(valid.sum())
            components = component_scores(probs, y)
            for j, q in enumerate(QUALITIES):
                mask = (y > 0) & ((y - 1) % len(QUALITIES) == j)
                frames[q] += [
                    mask.sum(),
                    (mask & (prediction == y)).sum(),
                    *components[mask].sum(axis=0),
                ]
            boundaries = np.array(
                [
                    ev[j][0]
                    for j in range(1, len(ev))
                    if ev[j - 1][2] >= 0 and ev[j][2] >= 0 and ev[j - 1][2] != ev[j][2]
                ]
            )
            distance = np.full(len(times), np.inf)
            for boundary in boundaries:
                distance = np.minimum(distance, abs(times - boundary))
            for tally, mask in (
                (near, valid & (distance <= 0.1)),
                (interior, valid & (distance > 0.1)),
            ):
                tally += [mask.sum(), ((prediction == y) & mask).sum()]
            for start, end, label in ev:
                if label < 0:
                    continue
                reference_total += 1
                mask = (times >= start) & (times < end)
                if not mask.any():
                    skipped += 1
                    continue
                pooled = [p[:, mask].mean(axis=1, keepdims=True) for p in probs]
                pred = int(decode_components(pooled)[0])
                correct_components = component_scores(pooled, np.array([label]))[0]
                event_rows.append(
                    [rid, start, end, label, pred, int(mask.sum()), int(label == pred)]
                )
                if label > 0:
                    q = QUALITIES[(label - 1) % len(QUALITIES)]
                    songs[q].add(rid)
                    events[q] += [1, int(label == pred), *correct_components]
    metrics = json.loads((run / "metrics.json").read_text())
    assert np.array_equal(frame_confusion, np.load(run / "predictions/confusion.npy"))
    assert abs(frame_confusion.trace() / frame_confusion.sum() - metrics["frame_accuracy"]) < 1e-12
    for unit, counts in (("frame", frames), ("event", events)):
        rows = []
        for q, values in counts.items():
            n, correct, root, triad, ext = map(int, values)
            rows.append(
                [
                    q,
                    n,
                    len(songs[q]),
                    correct,
                    correct / n if n else "",
                    root / n if n else "",
                    triad / n if n else "",
                    ext / n if n else "",
                ]
            )
        table(
            output / f"{unit}_qualities.csv",
            "quality support songs correct_complete_chords complete_chord_recall root_accuracy triad_accuracy extension_accuracy",
            rows,
        )
    table(output / "events.csv", "id start end reference prediction frames correct", event_rows)
    rare = [
        r["quality"]
        for r in csv.DictReader(Path("reports/p0/qualities.csv").open())
        if r["rare"] == "True"
    ]
    summary = dict(
        protocol="Mean per-head probabilities on merged reference intervals; frozen teacher decoder.",
        run_revision=metadata["git_revision"],
        reference_events=reference_total,
        scored_events=len(event_rows),
        events_without_frames=skipped,
        event_accuracy=sum(r[-1] for r in event_rows) / len(event_rows),
        rare_macro_event_recall=float(
            np.mean([events[q][1] / events[q][0] for q in rare if events[q][0]])
        ),
        rare_qualities=rare,
        gate_frame_accuracy=gate_correct / gate_total,
        component_frame_accuracy={
            name: float(sum(v[i] for v in frames.values()) / sum(v[0] for v in frames.values()))
            for name, i in (("root", 2), ("triad", 3), ("extension", 4))
        },
        near_boundary=dict(frames=int(near[0]), accuracy=float(near[1] / near[0])),
        interior=dict(frames=int(interior[0]), accuracy=float(interior[1] / interior[0])),
        near_boundary_seconds=0.1,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/p1-baseline-kaggle"))
    parser.add_argument("--labels", type=Path, default=Path("data/raw/full.tar.xz"))
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    diagnose(a.run, a.labels, a.output)
