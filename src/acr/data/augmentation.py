"""Training-only log-CQT transposition with matching complete-chord labels."""

import math
import torch


def transpose_batch(x, y, shifts, bins_per_semitone, floor):
    if x.ndim != 3 or y.shape[0] != x.shape[0] or len(shifts) != len(x):
        raise ValueError("Expected matched batch of CQT and labels")
    if bins_per_semitone < 1 or not math.isfinite(floor):
        raise ValueError("Invalid CQT configuration")

    shifted = x.clone()
    labels = y.clone()

    for i, semitones in enumerate(shifts):
        if int(semitones) != semitones:
            raise ValueError("Shifts must be integer semitones")

        semitones = int(semitones)
        bins = semitones * bins_per_semitone

        if abs(bins) >= x.shape[1]:
            raise ValueError("Shift exceeds feature range")
        if not bins:
            continue

        # Original cache is checked to contain no all-zero frames; these are padding.
        padding = (x[i] == 0).all(0)
        shifted[i].fill_(floor)

        if bins > 0:
            shifted[i, bins:] = x[i, :-bins]
        else:
            shifted[i, :bins] = x[i, -bins:]
        shifted[i, :, padding] = 0

        chord = y[i] > 0
        labels[i][chord] = (
            1 + (((y[i][chord] - 1) // 14 + semitones) % 12) * 14 + (y[i][chord] - 1) % 14
        )

    return shifted, labels
