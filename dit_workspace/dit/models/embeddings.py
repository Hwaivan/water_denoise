"""Dynamic rectangular sine-cosine positions and continuous/discrete time."""
import math
import torch
from torch import nn


def sinusoidal(values, dimension):
    frequencies = torch.exp(-math.log(10000) * torch.arange(
        dimension // 2, device=values.device, dtype=torch.float32) / (dimension // 2))
    angles = values.float().unsqueeze(-1) * frequencies
    return torch.cat([angles.cos(), angles.sin()], dim=-1)


def position_embedding(grid, dimension, device):
    if dimension % 4:
        raise ValueError("2D sine-cosine dimension must be divisible by four")
    frequency, time = torch.meshgrid(torch.arange(grid[0], device=device),
                                     torch.arange(grid[1], device=device), indexing="ij")
    return torch.cat([sinusoidal(frequency.flatten(), dimension // 2),
                      sinusoidal(time.flatten(), dimension // 2)], -1)[None]


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_size=256):
        super().__init__()
        self.frequency_size = frequency_size
        self.mlp = nn.Sequential(nn.Linear(frequency_size, hidden_size), nn.SiLU(),
                                 nn.Linear(hidden_size, hidden_size))

    def forward(self, time):
        return self.mlp(sinusoidal(time, self.frequency_size))
