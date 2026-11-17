"""Align one McGill Billboard recording with its detailed chord annotations."""

import argparse
import csv
import json
import math
from bisect import bisect_right
from html import escape
from pathlib import Path


def audit(recording, data_root, output):
    if len(recording) != 4 or not recording.isascii() or not recording.isdigit():
        raise ValueError("Use the four-digit Billboard recording ID.")
    labels = data_root / "billboard-2.0.1-lab" / recording / "full.lab"
    features = data_root / "billboard-2.0-chordino" / recording / "bothchroma.csv"
    events = []
    for line in labels.read_text().splitlines():
        if not line.strip():
            continue
        start, end, label = line.split()
        start, end = float(start), float(end)
        # LAB exports can disagree at a shared boundary by floating-point roundoff.
        if events and abs(start - events[-1][1]) <= 1e-9:
            previous_start, _, previous_label = events[-1]
            events[-1] = previous_start, start, previous_label
        if not (0 <= start < end < math.inf) or (events and start < events[-1][1]):
            raise ValueError(f"Invalid interval: {line}")
        events.append((start, end, label))
    if not events:
        raise ValueError("Empty annotation file")
    starts = [start for start, _, _ in events]
    rows = []
    with features.open(newline="") as stream:
        for number, row in enumerate(csv.reader(stream), 1):
            if len(row) != 26:
                raise ValueError(f"Expected metadata, seconds and 24 chroma bins at row {number}")
            time, *bins = map(float, row[1:])
            if not all(map(math.isfinite, [time, *bins])) or (rows and time <= rows[-1][0]):
                raise ValueError(f"Invalid feature or timestamp at row {number}")
            index = bisect_right(starts, time) - 1
            label = events[index][2] if index >= 0 and time < events[index][1] else ""
            rows.append((time, label, *bins))
    if not rows:
        raise ValueError("Empty chroma file")

    output.mkdir(parents=True, exist_ok=True)
    with (output / "frames.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "chord", *(f"chroma_{i}" for i in range(24))])
        writer.writerows(rows)

    # Show every reference interval above the actual frame timestamps on a common axis.
    duration = max(events[-1][1], rows[-1][0])
    scale = 1000 / duration
    palette = {label: (i * 137) % 360 for i, label in enumerate(sorted(set(e[2] for e in events)))}
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1020 150">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g transform="translate(10,0)"><text y="18">Reference chords (hover for labels)</text>']
    for start, end, label in events:
        svg.append(f'<rect x="{start*scale}" y="27" width="{(end-start)*scale}" height="28" '
                   f'fill="hsl({palette[label]},55%,55%)"><title>{escape(label)} '
                   f'{start:.3f}–{end:.3f} s</title></rect>')
    svg.append('<text y="78">Chroma frames (red = outside annotation)</text>')
    for time, label, *_ in rows:
        color = "#19756a" if label else "#da322c"
        svg.append(f'<path d="M{time*scale} 85v15" stroke="{color}" stroke-width=".4"/>')
    for second in range(0, int(duration) + 1, max(1, math.ceil(duration / 10))):
        svg.append(f'<text x="{second*scale}" y="126">{second}s</text>')
    (output / "timeline.svg").write_text("\n".join(svg + ["</g></svg>"]), encoding="utf-8")
    summary = dict(recording=recording, annotation_intervals=len(events), frames=len(rows),
                   labeled_frames=sum(bool(row[1]) for row in rows),
                   last_frame_s=rows[-1][0], annotation_end_s=events[-1][1])
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.recording, args.data_root, args.output), indent=2))
