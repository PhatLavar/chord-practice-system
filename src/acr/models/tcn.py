"""Compact offline TCN with root, triad, extension and no-chord heads."""

import torch
from torch import nn
from torch.nn import functional as F
from acr.vocabulary import TRIADS, EXTENSIONS


class FrameTeacher(nn.Module):
    def __init__(self, width=48, dilations=(1, 2, 4, 8, 16), input_channels=24):
        super().__init__()
        self.input = nn.Conv1d(input_channels, width, 1)
        self.blocks = nn.ModuleList(
            nn.Conv1d(width, width, 3, padding=d, dilation=d) for d in dilations
        )
        self.heads = nn.ModuleList(
            nn.Conv1d(width, size, 1) for size in (2, 12, len(TRIADS), len(EXTENSIONS))
        )
        self.context = sum(dilations)

    def encode_features(self, x):
        h = F.relu(self.input(x))

        for block in self.blocks:
            h = h + F.relu(block(h))

        return h

    def forward(self, x):
        h = self.encode_features(x)
        return [head(h) for head in self.heads]


class EventTeacher(FrameTeacher):
    def forward(self, x, pool_mask):
        if pool_mask.shape != (x.shape[0], x.shape[-1]) or not pool_mask.any(dim=1).all():
            raise ValueError("Every event needs at least one pooling frame")

        h = self.encode_features(x)
        mask = pool_mask.unsqueeze(1).to(h.dtype)
        pooled = (h * mask).sum(dim=2, keepdim=True) / mask.sum(dim=2, keepdim=True)

        return [head(pooled) for head in self.heads]
