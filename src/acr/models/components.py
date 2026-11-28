"""Factor targets, loss and legal-quality decoding shared by teacher backbones."""

import torch
from torch.nn import functional as F
from acr.vocabulary import QUALITIES, TRIADS, EXTENSIONS, COMPONENTS


def targets(labels):
    valid_chord = labels > 0
    quality = (labels.clamp_min(1) - 1) % len(QUALITIES)
    triads = labels.new_tensor([TRIADS.index(COMPONENTS[q][0]) for q in QUALITIES])
    extensions = labels.new_tensor([EXTENSIONS.index(COMPONENTS[q][1]) for q in QUALITIES])
    gate = torch.where(labels >= 0, (labels == 0).long(), -100)

    return [
        gate,
        torch.where(valid_chord, (labels - 1) // len(QUALITIES), -100),
        torch.where(valid_chord, triads[quality], -100),
        torch.where(valid_chord, extensions[quality], -100),
    ]


def losses(logits, labels, class_weights=None):
    result = []
    for index, (head, target) in enumerate(zip(logits, targets(labels))):
        mask = target != -100
        weight = class_weights[index] if class_weights is not None else None
        result.append(
            F.cross_entropy(head.transpose(1, 2)[mask], target[mask], weight=weight)
            if mask.any()
            else head.sum() * 0
        )

    return result


def decode(logits):
    gate, root, triad, extension = [h.log_softmax(1) for h in logits]

    # Gate odds choose N; supported chords maximize the joint component score.
    scores = torch.stack(
        [
            triad[:, TRIADS.index(COMPONENTS[q][0])]
            + extension[:, EXTENSIONS.index(COMPONENTS[q][1])]
            for q in QUALITIES
        ],
        1,
    )
    chord = 1 + root.argmax(1) * len(QUALITIES) + scores.argmax(1)

    return torch.where(gate[:, 1] > gate[:, 0], 0, chord)
