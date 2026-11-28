"""Frame-window datasets and quality/song-balanced sampling."""

import numpy as np
import torch
from torch.utils.data import Dataset
from acr.vocabulary import QUALITIES


class Windows(Dataset):
    def __init__(self, songs, length, context, include_padding_mask=False):
        self.songs, self.length, self.context = songs, length, context
        self.include_padding_mask = include_padding_mask
        self.windows = [
            (rid, start)
            for rid, song in songs.items()
            for start in range(0, len(song["y"]), length)
            if np.any(song["y"][start : start + length] >= 0)
        ]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        rid, start = self.windows[index]
        return self.get_window(rid, start)

    def get_window(self, rid, start):
        song = self.songs[rid]
        stop = min(start + self.length, len(song["y"]))

        x = np.zeros((song["x"].shape[1], self.length + 2 * self.context), dtype=np.float32)
        lo, hi = max(0, start - self.context), min(
            len(song["y"]), start + self.length + self.context
        )
        offset = lo - start + self.context
        x[:, offset : offset + hi - lo] = song["x"][lo:hi].T

        y = np.full(self.length, -100, dtype=np.int64)
        y[: stop - start] = song["y"][start:stop]

        if self.include_padding_mask:
            padding = np.ones(x.shape[-1], dtype=bool)
            padding[offset : offset + hi - lo] = False
            return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(padding)

        return torch.from_numpy(x), torch.from_numpy(y)


class QualityBalancedWindows(Windows):
    """Half natural windows, half quality/song-balanced centered windows."""

    def __init__(self, songs, length, context, include_padding_mask=False):
        super().__init__(songs, length, context, include_padding_mask)
        self.anchors = {}
        for rid, song in songs.items():
            y = song["y"]
            for quality in range(len(QUALITIES)):
                indices = np.flatnonzero((y > 0) & ((y - 1) % len(QUALITIES) == quality))
                if len(indices):
                    self.anchors.setdefault(quality, []).append((rid, indices))

        self.qualities = sorted(self.anchors)
        if not self.qualities:
            raise ValueError("Balanced sampling requires observed chord qualities")

    def __getitem__(self, index):
        if torch.rand(()).item() < 0.5:
            return super().__getitem__(index)

        quality = self.qualities[torch.randint(len(self.qualities), ()).item()]
        entries = self.anchors[quality]
        rid, indices = entries[torch.randint(len(entries), ()).item()]
        anchor = int(indices[torch.randint(len(indices), ()).item()])
        start = max(0, min(anchor - self.length // 2, len(self.songs[rid]["y"]) - self.length))

        return self.get_window(rid, start)
