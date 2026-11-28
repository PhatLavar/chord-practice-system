"""Compare four independent F/E combinations on identical Billboard gold intervals."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from acr.evaluation.confidence import (
    HEADS,
    VALUES,
    THRESHOLDS,
    read_rows,
    write_rows,
    tally,
    decisions,
)
from acr.evaluation.frame import probabilities, decode_components
from acr.vocabulary import QUALITIES
from acr.models.components import targets
from acr.training.event import compare, FIELDS
from acr.data.jaah import digest


def load_result(directory, teacher):
    rows = read_rows(directory / "events.csv")
    meta = json.loads((directory / "run.json").read_text())
    keys = [(r["id"], float(r["start"]), float(r["end"]), int(r["reference"])) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate event keys")
    path = directory / ("event_probabilities.npz" if teacher == "F" else "event_logits.npz")
    with np.load(path) as data:
        y = data["reference"]
        saved = data["prediction"]
        probs = [
            data[f"probabilities_{j}"] if teacher == "F" else probabilities(data[f"logits_{j}"])
            for j in range(4)
        ]
    if not np.array_equal(y, [k[-1] for k in keys]):
        raise ValueError("Reference labels differ")
    for j, p in enumerate(probs):
        if (
            p.shape != (len(VALUES[j]), len(y))
            or not np.isfinite(p).all()
            or (p < 0).any()
            or not np.allclose(p.sum(0), 1)
        ):
            raise ValueError("Invalid event probabilities")
        if not np.array_equal(p.argmax(0), [int(r[HEADS[j]]) for r in rows]):
            raise ValueError("Head decisions differ")
    predicted = decode_components(probs)
    if not np.array_equal(predicted, saved) or not np.array_equal(
        predicted, [int(r["prediction"]) for r in rows]
    ):
        raise ValueError("Decoded predictions differ")
    return (
        keys,
        y,
        probs,
        rows,
        meta,
        dict(
            run_json=digest(directory / "run.json"),
            events_csv=digest(directory / "events.csv"),
            probabilities=digest(path),
        ),
    )


def complete_scores(y, predicted, rare):
    rows = []
    for j, q in enumerate(QUALITIES):
        mask = (y > 0) & ((y - 1) % 14 == j)
        chosen = (predicted > 0) & ((predicted - 1) % 14 == j)
        n = int(mask.sum())
        correct = int((mask & (predicted == y)).sum())
        count = int(chosen.sum())
        rows.append(
            dict(
                quality=q,
                support=n,
                predicted=count,
                correct_complete=correct,
                recall=correct / n if n else None,
                precision=correct / count if count else None,
            )
        )
    return (
        dict(
            event_accuracy=float((y == predicted).mean()),
            rare_macro_event_recall=(
                float(
                    np.mean(
                        [
                            r["recall"]
                            for r in rows
                            if r["quality"] in rare and r["recall"] is not None
                        ]
                    )
                )
                if any(r["quality"] in rare and r["recall"] is not None for r in rows)
                else None
            ),
            macro_quality_event_recall=float(
                np.mean([r["recall"] for r in rows if r["recall"] is not None])
            ),
        ),
        rows,
    )


def summarize_pair(f, e, y, songs, rare):
    fp, ep = decode_components(f), decode_components(e)
    raremask = np.isin(np.array(["N" if v == 0 else QUALITIES[(v - 1) % 14] for v in y]), rare)
    summary = {}
    for name, mask in [("all", np.ones(len(y), dtype=bool)), ("true_rare", raremask)]:
        a, b = fp == y, ep == y
        summary[name] = dict(
            events=int(mask.sum()),
            both_correct=int((mask & a & b).sum()),
            F_only=int((mask & a & ~b).sum()),
            E_only=int((mask & ~a & b).sum()),
            both_wrong=int((mask & ~a & ~b).sum()),
        )
        agreement = tally(y, fp, np.zeros(len(y)), fp == ep, mask, songs)
        agreement.pop("mean_confidence")
        summary[name]["complete_agreement"] = agreement
    return summary


def run(root, output, smoke=False):
    if output.exists():
        raise FileExistsError("Use a fresh pair-analysis directory")
    rare = [
        r["quality"] for r in read_rows(Path("reports/p0/qualities.csv")) if r["rare"] == "True"
    ]
    loaded = {}
    provenance = {}
    expected = None
    first = None
    for teacher in ("F", "E"):
        for arm in ("billboard_control", "jaah_mixed"):
            name = f"{teacher}_{arm}"
            directory = root / teacher / arm
            result = load_result(directory, teacher)
            keys, y, probs, rows, meta, hashes = result
            if bool(meta["smoke"]) != smoke:
                raise ValueError("Cannot mix smoke and full results")
            if meta["evaluation_partition"] != (
                "validation_subset_smoke" if smoke else "validation"
            ):
                raise ValueError("Wrong evaluation partition")
            if meta["arm"] != arm:
                raise ValueError("Wrong experimental arm")
            if expected is None:
                expected = keys
                first = meta
            if keys != expected:
                raise ValueError("Event keys/order mismatch between teachers")
            if (
                meta["data"] != first["data"]
                or meta["auxiliary"]["source_sha256"] != first["auxiliary"]["source_sha256"]
                or meta["config"]["seed"] != first["config"]["seed"]
            ):
                raise ValueError("Incompatible data or seed")
            loaded[name] = result
            provenance[name] = hashes
    if not smoke:
        frozen = read_rows(Path("reports/p2/f-reference/events.csv"))
        if expected != [
            (r["id"], float(r["start"]), float(r["end"]), int(r["reference"])) for r in frozen
        ]:
            raise ValueError("Results differ from frozen validation events")
    for teacher in ("F", "E"):
        a = loaded[f"{teacher}_billboard_control"][4]
        b = loaded[f"{teacher}_jaah_mixed"][4]
        if (
            a["config"] != b["config"]
            or a["steps_per_epoch"] != b["steps_per_epoch"]
            or a["epochs"] != b["epochs"]
            or a["train_ids"] != b["train_ids"]
        ):
            raise ValueError("Unmatched control/mixed training protocol")
    output.mkdir(parents=True)
    individuals = []
    for name, (_, y, probs, _, _, _) in loaded.items():
        metrics, qualities = complete_scores(y, decode_components(probs), rare)
        individuals.append(dict(run=name, **metrics))
        write_rows(output / f"{name}_qualities.csv", qualities)
    write_rows(output / "individuals.csv", individuals)
    songs = np.array([k[0] for k in expected])
    truth = [t.numpy() for t in targets(torch.from_numpy(y))]
    qualities = np.array(["N" if v == 0 else QUALITIES[(v - 1) % 14] for v in y])
    summaries = {}
    curves = []
    classes = []
    slices = []
    for fa in ("billboard_control", "jaah_mixed"):
        for ea in ("billboard_control", "jaah_mixed"):
            name = f"F_{fa}__E_{ea}"
            f = loaded[f"F_{fa}"][2]
            e = loaded[f"E_{ea}"][2]
            directory = output / name
            directory.mkdir()
            e_rows = [[r[k] for k in FIELDS] for r in loaded[f"E_{ea}"][3]]
            # compare() expects numeric timestamps from evaluate(), not CSV strings.
            for r in e_rows:
                r[1], r[2] = float(r[1]), float(r[2])
            compare(root / "F" / fa / "events.csv", e_rows, directory)
            summaries[name] = summarize_pair(f, e, y, songs, rare)
            for j, head in enumerate(HEADS):
                for method in ("F", "E", "agreement"):
                    for threshold in THRESHOLDS:
                        pred, conf, mask = decisions(f, e, j, method, threshold, True)
                        prefix = dict(pair=name, head=head, method=method, threshold=threshold)
                        eligible = np.ones(len(y), dtype=bool)
                        curves.append(
                            dict(**prefix, **tally(truth[j], pred, conf, mask, eligible, songs))
                        )
                        for value, label in enumerate(VALUES[j]):
                            stats = tally(
                                truth[j], pred, conf, mask & (pred == value), eligible, songs
                            )
                            support = int((truth[j] == value).sum())
                            classes.append(
                                dict(
                                    **prefix,
                                    value=label,
                                    support=support,
                                    accepted_recall=stats["correct"] / support if support else None,
                                    **stats,
                                )
                            )
                        for q in ["N", *QUALITIES, "ALL_RARE"]:
                            subset = np.isin(qualities, rare) if q == "ALL_RARE" else qualities == q
                            slices.append(
                                dict(
                                    **prefix,
                                    true_quality=q,
                                    **tally(truth[j], pred, conf, mask, subset, songs),
                                )
                            )
    for name, rows in [
        ("curves", curves),
        ("predicted_classes", classes),
        ("true_qualities", slices),
    ]:
        write_rows(output / f"{name}.csv", rows)
    summary = dict(
        seed=first["config"]["seed"],
        smoke=smoke,
        events=len(y),
        pairs=summaries,
        provenance=provenance,
        protocol="Same gold intervals for all four pairs. F averages frame probabilities; E uses event softmax. Operational confidence thresholds are exploratory, not calibrated.",
        caveat=(
            "Smoke only; no performance conclusion."
            if smoke
            else "One seed on reused validation. No pseudo-label acceptance or held-out-test claim."
        ),
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(events=len(y), smoke=smoke, pairs=len(summaries), individuals=individuals),
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    run(a.runs, a.output, a.smoke)
