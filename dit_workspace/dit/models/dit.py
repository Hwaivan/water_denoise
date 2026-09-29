"""One Conditional DiT backbone for score, noise, or velocity prediction."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .embeddings import TimestepEmbedder, position_embedding
from .blocks import DiTBlock, FinalLayer


class InputAdapter(nn.Module):
    """Aligned spectral condition; future adapters can replace this boundary."""
    def forward(self, state, condition):
        if state.shape != condition.shape or state.ndim != 4 or torch.is_complex(state):
            raise ValueError("State and condition must be aligned real [B,C,F,T]")
        return torch.cat([state, condition], dim=1)


class PatchEmbed(nn.Module):
    def __init__(self, input_channels, dimension, patch_size):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.projection = nn.Conv2d(input_channels, dimension, self.patch_size, self.patch_size)

    def forward(self, x):
        original = x.shape[-2:]
        pf, pt = self.patch_size
        x = F.pad(x, (0, -original[1] % pt, 0, -original[0] % pf))
        patches = self.projection(x)
        return patches.flatten(2).transpose(1, 2), patches.shape[-2:], original


class ConditionalDiT(nn.Module):
    def __init__(self, channels=2, depth=12, hidden_size=384, num_heads=6,
                 mlp_ratio=4, patch_size=(4, 8), dropout=0., adaLN_zero=True,
                 position_embedding="sincos_2d", name="dit", max_tokens=8192):
        super().__init__()
        if name != "dit" or not adaLN_zero or position_embedding != "sincos_2d":
            raise ValueError("Only DiT with adaLN-Zero and sincos_2d is implemented")
        if hidden_size % 4 or hidden_size % num_heads or depth < 1 or min(patch_size) < 1:
            raise ValueError("Invalid dimension, heads, depth or patch size")
        self.channels, self.hidden_size = channels, hidden_size
        self.patch_size, self.max_tokens = tuple(patch_size), int(max_tokens)
        self.input_adapter = InputAdapter()
        self.patch_embed = PatchEmbed(channels * 2, hidden_size, patch_size)
        self.time_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, channels)
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.patch_embed.projection.weight.flatten(1))
        nn.init.zeros_(self.patch_embed.projection.bias)
        for module in (self.time_embedder.mlp[0], self.time_embedder.mlp[2]):
            nn.init.normal_(module.weight, std=.02)
        for module in [block.modulation[-1] for block in self.blocks] + [
                self.final_layer.modulation[-1], self.final_layer.projection]:
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def geometry(self, frequency_bins, frames):
        grid = [math.ceil(frequency_bins / self.patch_size[0]), math.ceil(frames / self.patch_size[1])]
        return {"patch_grid": grid, "token_count": math.prod(grid),
                "total_parameters": sum(p.numel() for p in self.parameters())}

    def unpatchify(self, patches, grid, original):
        b = patches.shape[0]
        pf, pt = self.patch_size
        x = patches.reshape(b, grid[0], grid[1], pf, pt, self.channels)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, self.channels, grid[0]*pf, grid[1]*pt)
        return x[..., :original[0], :original[1]]

    def forward(self, state, condition, time):
        if state.shape[1] != self.channels or time.shape != (state.shape[0],):
            raise ValueError("Wrong channel or time shape")
        if self.geometry(*state.shape[-2:])["token_count"] > self.max_tokens:
            raise ValueError("Attention token budget exceeded; reduce chunk_seconds or crop_frames")
        tokens, grid, original = self.patch_embed(self.input_adapter(state, condition))
        tokens = tokens + position_embedding(grid, self.hidden_size, tokens.device).to(tokens.dtype)
        embedding = self.time_embedder(time)
        for block in self.blocks:
            tokens = block(tokens, embedding)
        return self.unpatchify(self.final_layer(tokens, embedding), grid, original)
