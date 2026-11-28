"""Audit saved player-held-out F/E components on supported gold events."""

import argparse
import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import soundfile as sf
import torch

from acr.data.guitarset.labels import map_symbol
from acr.vocabulary import QUALITIES, TRIADS, EXTENSIONS
from acr.models.components import decode, targets
from acr.vocabulary import chord_label


from collections import defaultdict
from torch.utils.data import DataLoader
from acr.data.events import Events, collate_events
from acr.data.guitarset.cqt import event_list
from acr.evaluation.intervals import score


def timeline(events, duration, labels, fill):
    intervals = []
    result = []
    previous = 0.0
    for e, label in zip(events, labels):
        start, end = float(e["time"]), float(e["time"] + e["duration"])
        # GuitarSet decimal endpoints can exceed WAV duration by <1 original sample.
        if start < previous - 1e-8 or end > duration + 1 / 44100:
            raise ValueError("Invalid reference coverage")
        end = min(end, duration)
        if end <= start:
            raise ValueError("Empty reference interval after clipping")
        if start > previous + 1e-8:
            intervals.append([previous, start])
            result.append(fill)
        intervals.append([start, end])
        result.append(label)
        previous = end
    if previous < duration:
        intervals.append([previous, duration])
        result.append(fill)
    return intervals, result


@torch.no_grad()
def evaluate(model, teacher, songs, references, device, batch_size, output=None):
    model.eval()
    events = event_list(songs, references)
    predicted = {}
    probabilities = {}
    if teacher == "F":
        for rid, s in songs.items():
            x = torch.from_numpy(s["x"].T.copy()).unsqueeze(0).to(device)
            # Match the zero-normalized-input context used by the training windows.
            x = torch.nn.functional.pad(x, (model.context, model.context))
            heads = [h[:, :, model.context : model.context + len(s["times"])] for h in model(x)]
            predicted[rid] = decode(heads)[0].cpu().numpy()
            probabilities[rid] = [h.softmax(1)[0].cpu().numpy() for h in heads]
            if output:
                np.savez_compressed(
                    output / (rid + ".npz"),
                    prediction=predicted[rid],
                    times=s["times"],
                    **{f"logits_{j}": h[0].cpu().numpy() for j, h in enumerate(heads)},
                )
    else:
        dataset = Events(songs, events, context=model.context)
        parts = [[] for _ in range(4)]
        for x, mask, _ in DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_events
        ):
            for dest, h in zip(parts, model(x.to(device), mask.to(device))):
                dest.append(h.cpu())
        heads = [torch.cat(p) for p in parts]
        labels = decode(heads).reshape(-1).numpy()
        if output:
            np.savez_compressed(
                output / "event_logits.npz",
                prediction=labels,
                **{f"logits_{j}": h[:, :, 0].numpy() for j, h in enumerate(heads)},
            )
        for e, label in zip(events, labels):
            predicted.setdefault(e[0], []).append(int(label))
    totals = {}
    per_song = {}
    correct = count = 0
    quality = defaultdict(lambda: dict(events=0, correct=0, predicted=0))
    for rid, s in songs.items():
        reference = references[rid]
        raw = reference["events"]
        duration = reference["duration"]
        ref_i, ref_l = timeline(raw, duration, [e["value"] for e in raw], "X")
        if teacher == "F":
            times = s["times"]
            est_i = np.stack([times, np.r_[times[1:], duration]], axis=1)
            est_l = [chord_label(int(v)) for v in predicted[rid]]
            event_predictions = []
            for e in raw:
                lo, hi = np.searchsorted(times, [e["time"], e["time"] + e["duration"]])
                pooled = [
                    torch.from_numpy(p[:, lo:hi].mean(1)).clamp_min(1e-30).log()[None, :, None]
                    for p in probabilities[rid]
                ]
                event_predictions.append(int(decode(pooled).item()))
        else:
            event_predictions = predicted[rid]
            est_i, est_l = timeline(raw, duration, [chord_label(v) for v in event_predictions], "N")
        result = score(ref_i, ref_l, est_i, est_l)
        per_song[rid] = result
        if output:
            # Preserve float64 endpoints, including tiny positive trailing gaps.
            (output / (rid + ".lab")).write_text(
                "".join(f"{a:.17g} {b:.17g} {label}\n" for (a, b), label in zip(est_i, est_l))
            )
        for name, values in result["metrics"].items():
            bucket = totals.setdefault(
                name, dict(correct_seconds=0.0, eligible_seconds=0.0, excluded_seconds=0.0)
            )
            for k in bucket:
                bucket[k] += values[k]
        for e, pred in zip(raw, event_predictions):
            target = map_symbol(e["value"])["complete"]
            if target < 0:
                continue
            correct += int(pred == target)
            count += 1
            q = "N" if target == 0 else QUALITIES[(target - 1) % 14]
            quality[q]["events"] += 1
            quality[q]["correct"] += int(pred == target)
            pq = "N" if pred == 0 else QUALITIES[(pred - 1) % 14]
            quality[pq]["predicted"] += 1
    for values in totals.values():
        values["score"] = (
            values["correct_seconds"] / values["eligible_seconds"]
            if values["eligible_seconds"]
            else None
        )
    result = dict(
        teacher=teacher,
        metrics=totals,
        supported_oracle_event_accuracy=correct / count if count else None,
        supported_events=count,
        quality_supported_scope=dict(quality),
        protocol="Full audio duration with raw performed labels; gaps X. F predicted frame intervals; E gold intervals. Event accuracy restricted to complete-supported labels.",
    )
    if output:
        (output / "per_song.json").write_text(json.dumps(per_song, indent=2) + "\n")
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(root, out, references, clips=None):
    out.mkdir(exist_ok=False, parents=True)
    refs = json.loads(references.read_text())
    labels = [["chord", "N"], list(range(12)), TRIADS, EXTENSIONS]
    records = []
    summary = {}
    matrices = []
    for teacher in ["F", "E"]:
        offset = 0
        truth = []
        heads = [[], [], [], []]
        predictions = []
        if teacher == "E":
            with np.load(root / teacher / "predictions/event_logits.npz") as data:
                allprob = [torch.from_numpy(data[f"logits_{j}"]).softmax(1) for j in range(4)]
        for rid, ref in sorted(refs.items()):
            if teacher == "F":
                with np.load(root / teacher / "predictions" / (rid + ".npz")) as data:
                    times = data["times"]
                    frame = [torch.from_numpy(data[f"logits_{j}"]).softmax(0) for j in range(4)]
            for e in ref["events"]:
                if teacher == "F":
                    lo, hi = np.searchsorted(times, [e["time"], e["time"] + e["duration"]])
                    assert hi > lo
                    prob = [p[:, lo:hi].mean(1) for p in frame]
                else:
                    prob = [p[offset] for p in allprob]
                offset += 1
                target = map_symbol(e["value"])["complete"]
                if target <= 0:
                    continue
                predicted = int(
                    decode([p.clamp_min(1e-30).log()[None, :, None] for p in prob]).item()
                )
                truth.append(target)
                predictions.append(predicted)
                components = [int(p.argmax()) for p in prob]
                for dest, c in zip(heads, components):
                    dest.append(c)
                records.append(
                    dict(
                        teacher=teacher,
                        recording=rid,
                        start=e["time"],
                        end=e["time"] + e["duration"],
                        reference=e["value"],
                        target=chord_label(target),
                        prediction=chord_label(predicted),
                        root_confidence=float(prob[1].max()),
                        triad_confidence=float(prob[2].max()),
                        extension_confidence=float(prob[3].max()),
                        extension_prediction=EXTENSIONS[components[3]],
                    )
                )
        target_heads = targets(torch.tensor(truth))
        counts = {}
        for j, name in enumerate(["gate", "root", "triad", "extension"]):
            actual = target_heads[j].numpy()
            pred = np.array(heads[j])
            matrix = np.zeros((len(labels[j]), len(labels[j])), dtype=int)
            np.add.at(matrix, (actual, pred), 1)
            assert int(matrix.sum()) == len(truth)
            for a, row in enumerate(matrix):
                for b, count in enumerate(row):
                    matrices.append(
                        dict(
                            teacher=teacher,
                            head=name,
                            reference=labels[j][a],
                            prediction=labels[j][b],
                            events=int(count),
                        )
                    )
            counts[name] = dict(correct=int((actual == pred).sum()), events=len(actual))
        actual_ext = target_heads[3].numpy()
        actual_root = target_heads[1].numpy()
        actual_triad = target_heads[2].numpy()
        pred = np.array(predictions)
        correct_root = np.array(heads[1]) == actual_root
        correct_triad = np.array(heads[2]) == actual_triad
        correct_ext = np.array(heads[3]) == actual_ext
        extended = actual_ext != EXTENSIONS.index("none")
        saved = json.loads((root / teacher / "metrics.json").read_text())
        assert len(truth) == saved["supported_events"]
        assert abs(float(np.mean(pred == truth)) - saved["supported_oracle_event_accuracy"]) < 1e-12
        summary[teacher] = dict(
            components=counts,
            supported_events=len(truth),
            extended_events=int(extended.sum()),
            extended_predicted_none=int(
                (extended & (np.array(heads[3]) == EXTENSIONS.index("none"))).sum()
            ),
            extension_correct_but_complete_wrong=int((correct_ext & (pred != truth)).sum()),
            root_and_triad_correct_extension_wrong=int(
                (correct_root & correct_triad & ~correct_ext).sum()
            ),
            raw_all_heads_correct=int((correct_root & correct_triad & correct_ext).sum()),
        )
    write_csv(out / "confusion.csv", matrices)
    write_csv(out / "events.csv", records)
    # Paired examples are selected deterministically, before any listening judgment.
    frows = [r for r in records if r["teacher"] == "F"]
    erows = [r for r in records if r["teacher"] == "E"]
    assert [(r["recording"], r["start"]) for r in frows] == [
        (r["recording"], r["start"]) for r in erows
    ]
    selected = []
    for quality in ["7", "maj7", "min7", "maj6", "min6"]:
        candidates = [
            (f, e)
            for f, e in zip(frows, erows)
            if f["target"].split(":")[1] == quality
            and f["prediction"] != f["target"]
            and e["prediction"] != e["target"]
        ]
        if candidates:
            f, e = candidates[0]
            selected.append(
                dict(
                    recording=f["recording"],
                    start=f["start"],
                    end=f["end"],
                    reference=f["reference"],
                    F=f["prediction"],
                    E=e["prediction"],
                )
            )
    if clips is not None:
        clips.mkdir(exist_ok=False)
        with zipfile.ZipFile("data/raw/guitarset/audio_mono-mic.zip") as z:
            names = {
                Path(n).stem.removesuffix("_mic"): n
                for n in z.namelist()
                if n.endswith(".wav") and not n.startswith("__MACOSX")
            }
            for i, row in enumerate(selected):
                audio, sr = sf.read(io.BytesIO(z.read(names[row["recording"]])), dtype="float32")
                start = max(0, row["start"] - 1)
                end = min(len(audio) / sr, row["end"] + 1)
                path = clips / f"{i+1:02d}.wav"
                sf.write(path, audio[int(start * sr) : int(end * sr)], sr, subtype="PCM_16")
                row.update(clip=path.as_posix(), clip_start=start, clip_end=end)
    if selected:
        write_csv(out / "review_candidates.csv", selected)
    summary["scope"] = (
        "Supported chord gold events only; F mean frame probabilities, E event softmax. Raw head argmax confusion precedes legal quality decoder. Clips are review candidates, not independently verified chord judgments."
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("runs/p2-guitarset-cqt-player"))
    p.add_argument(
        "--output", type=Path, default=Path("reports/p2/guitarset-cqt/player-diagnostics")
    )
    p.add_argument("--references", type=Path)
    p.add_argument(
        "--clips", type=Path, help="Optionally export review audio into a fresh directory"
    )
    a = p.parse_args()
    run(a.root, a.output, a.references or a.root / "validation_references.json", a.clips)
