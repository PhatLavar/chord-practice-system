"""Compact offline TCN with root, triad, extension and no-chord heads."""
import torch
from torch import nn
from torch.nn import functional as F
from p1_data import QUALITIES, TRIADS, EXTENSIONS
from corpus import COMPONENTS


class FrameTeacher(nn.Module):
    def __init__(self, width=48, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        self.input = nn.Conv1d(24, width, 1)
        self.blocks = nn.ModuleList(nn.Conv1d(width, width, 3, padding=d, dilation=d) for d in dilations)
        self.heads = nn.ModuleList(nn.Conv1d(width, size, 1) for size in (2, 12, len(TRIADS), len(EXTENSIONS)))
        self.context = sum(dilations)

    def forward(self, x):
        h = F.relu(self.input(x))
        for block in self.blocks:
            h = h + F.relu(block(h))
        return [head(h) for head in self.heads]


def targets(labels):
    valid_chord = labels > 0
    quality = (labels.clamp_min(1) - 1) % len(QUALITIES)
    triads = labels.new_tensor([TRIADS.index(COMPONENTS[q][0]) for q in QUALITIES])
    extensions = labels.new_tensor([EXTENSIONS.index(COMPONENTS[q][1]) for q in QUALITIES])
    gate = torch.where(labels >= 0, (labels == 0).long(), -100)
    return [gate, torch.where(valid_chord, (labels-1)//len(QUALITIES), -100),
            torch.where(valid_chord, triads[quality], -100),
            torch.where(valid_chord, extensions[quality], -100)]


def losses(logits, labels):
    result = []
    for head, target in zip(logits, targets(labels)):
        mask = target != -100
        result.append(F.cross_entropy(head.transpose(1,2)[mask], target[mask])
                      if mask.any() else head.sum()*0)
    return result


def decode(logits):
    gate, root, triad, extension = [h.log_softmax(1) for h in logits]
    # Gate odds choose N; supported chords maximize the joint component score.
    scores = torch.stack([triad[:,TRIADS.index(COMPONENTS[q][0])] +
                          extension[:,EXTENSIONS.index(COMPONENTS[q][1])] for q in QUALITIES],1)
    chord = 1 + root.argmax(1)*len(QUALITIES) + scores.argmax(1)
    return torch.where(gate[:,1] > gate[:,0], 0, chord)
