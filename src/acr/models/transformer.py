"""Project-specific compact bidirectional Transformer; component heads match F."""
import math
import torch
from torch import nn
from acr.vocabulary import TRIADS, EXTENSIONS


class TransformerFrameTeacher(nn.Module):
    def __init__(self, width=48, window=512):
        super().__init__()
        self.context = 31
        self.window = window
        self.input = nn.Linear(24, width)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(width, 4, dim_feedforward=2*width,
                dropout=0., activation="gelu", batch_first=True, norm_first=True)
            for _ in range(2)])
        self.norm = nn.LayerNorm(width)
        self.heads = nn.ModuleList(nn.Linear(width, size) for size in (2,12,len(TRIADS),len(EXTENSIONS)))
        positions = torch.arange(window+2*self.context).float().unsqueeze(1)
        scales = torch.exp(torch.arange(0,width,2).float()*(-math.log(10000.)/width))
        encoding = torch.zeros(window+2*self.context,width)
        encoding[:,0::2] = torch.sin(positions*scales)
        encoding[:,1::2] = torch.cos(positions*scales)
        self.register_buffer("positions",encoding)

    def forward(self, x, padding_mask):
        if x.shape[-1] > len(self.positions) or padding_mask.shape != (x.shape[0],x.shape[-1]):
            raise ValueError("Unexpected attention window or padding mask")
        if padding_mask.all(dim=1).any():
            raise ValueError("Attention window has no real frames")
        h = self.input(x.transpose(1,2)) + self.positions[:x.shape[-1]]
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=padding_mask)
        h = self.norm(h)
        return [head(h).transpose(1,2) for head in self.heads]
