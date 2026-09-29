"""Ordinary real-valued attention; timestep-conditioned adaLN-Zero blocks."""
import torch
from torch import nn
from torch.nn import functional as F


def modulate(tokens, shift, scale):
    return tokens * (1 + scale[:, None]) + shift[:, None]


class Attention(nn.Module):
    def __init__(self, dimension, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.qkv = nn.Linear(dimension, dimension * 3)
        self.projection = nn.Linear(dimension, dimension)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.)
        return self.projection(attended.transpose(1, 2).reshape(b, n, d))


class DiTBlock(nn.Module):
    def __init__(self, dimension, heads, mlp_ratio=4, dropout=0.):
        super().__init__()
        self.norm_attention = nn.LayerNorm(dimension, elementwise_affine=False, eps=1e-6)
        self.norm_mlp = nn.LayerNorm(dimension, elementwise_affine=False, eps=1e-6)
        self.attention = Attention(dimension, heads, dropout)
        self.mlp = nn.Sequential(nn.Linear(dimension, int(dimension * mlp_ratio)),
                                 nn.GELU(approximate="tanh"), nn.Dropout(dropout),
                                 nn.Linear(int(dimension * mlp_ratio), dimension), nn.Dropout(dropout))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dimension, 6 * dimension))

    def forward(self, tokens, time_embedding):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(time_embedding).chunk(6, -1)
        tokens = tokens + gate_a[:, None] * self.attention(modulate(self.norm_attention(tokens), shift_a, scale_a))
        return tokens + gate_m[:, None] * self.mlp(modulate(self.norm_mlp(tokens), shift_m, scale_m))


class FinalLayer(nn.Module):
    def __init__(self, dimension, patch_size, channels):
        super().__init__()
        self.norm = nn.LayerNorm(dimension, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dimension, 2 * dimension))
        self.projection = nn.Linear(dimension, patch_size[0] * patch_size[1] * channels)

    def forward(self, tokens, time_embedding):
        shift, scale = self.modulation(time_embedding).chunk(2, -1)
        return self.projection(modulate(self.norm(tokens), shift, scale))
