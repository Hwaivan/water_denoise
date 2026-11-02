"""Conditional DiT: one spectral time frame per token, optional Mel condition."""
import torch
from torch import nn
from .dit import InputAdapter
from .blocks import DiTBlock, modulate
from .embeddings import TimestepEmbedder, sinusoidal


class FrameEmbed(nn.Module):
    def __init__(self, input_channels, frequency_bins, dimension, aux_condition_bins=0):
        super().__init__()
        self.aux_condition_bins = int(aux_condition_bins)
        self.projection = nn.Linear(input_channels * frequency_bins + self.aux_condition_bins, dimension)

    def forward(self, x, aux_condition=None):
        tokens = x.permute(0, 3, 1, 2).flatten(2)
        if self.aux_condition_bins:
            if (aux_condition is None or aux_condition.ndim != 4
                    or aux_condition.shape != (x.shape[0], 1, self.aux_condition_bins, x.shape[-1])):
                raise ValueError("Expected [B,1,M,T] aligned Mel condition")
            aux = aux_condition.permute(0, 3, 1, 2).flatten(2)
            tokens = torch.cat((tokens, aux), dim=-1)
        elif aux_condition is not None:
            raise ValueError("Unexpected auxiliary condition; configure aux_condition_bins")
        return self.projection(tokens)


class FrameFinalLayer(nn.Module):
    def __init__(self, dimension, channels, frequency_bins):
        super().__init__()
        self.norm = nn.LayerNorm(dimension, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dimension, 2 * dimension))
        self.projection = nn.Linear(dimension, channels * frequency_bins)

    def forward(self, tokens, time_embedding):
        shift, scale = self.modulation(time_embedding).chunk(2, -1)
        return self.projection(modulate(self.norm(tokens), shift, scale))


class FrameConditionalDiT(nn.Module):
    def __init__(self, frequency_bins, channels=2, depth=12, hidden_size=384,
                 num_heads=6, mlp_ratio=4, dropout=0., adaLN_zero=True,
                 position_embedding="sincos_1d", name="dit", max_tokens=8192,
                 aux_condition_bins=0):
        super().__init__()
        if name != "dit" or not adaLN_zero or position_embedding != "sincos_1d":
            raise ValueError("Frame DiT requires adaLN-Zero and sincos_1d")
        if (hidden_size < 2 or hidden_size % 2 or num_heads < 1
                or hidden_size % num_heads or depth < 1 or channels < 1
                or frequency_bins < 1 or max_tokens < 1 or aux_condition_bins < 0):
            raise ValueError("Invalid frame DiT dimensions or token budget")
        self.channels, self.frequency_bins = channels, frequency_bins
        self.hidden_size, self.max_tokens = hidden_size, int(max_tokens)
        self.aux_condition_bins = int(aux_condition_bins)
        self.input_adapter = InputAdapter()
        self.frame_embed = FrameEmbed(2 * channels, frequency_bins, hidden_size, aux_condition_bins)
        self.time_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.final_layer = FrameFinalLayer(hidden_size, channels, frequency_bins)
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in (self.time_embedder.mlp[0], self.time_embedder.mlp[2]):
            nn.init.normal_(module.weight, std=.02)
        for module in [block.modulation[-1] for block in self.blocks] + [
                self.final_layer.modulation[-1], self.final_layer.projection]:
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def geometry(self, frequency_bins, frames):
        if frequency_bins != self.frequency_bins or frames < 1:
            raise ValueError("Frame geometry requires configured frequency bins and positive frames")
        return {"tokenization": "frame1d", "frequency_bins": frequency_bins,
                "frame_count": frames, "token_count": frames,
                "aux_condition_bins": self.aux_condition_bins,
                "total_parameters": sum(p.numel() for p in self.parameters())}

    def forward(self, state, condition, time, aux_condition=None):
        aligned = self.input_adapter(state, condition)
        if state.shape[1] != self.channels or time.shape != (state.shape[0],):
            raise ValueError("Wrong channel or time shape")
        frames = state.shape[-1]
        if self.geometry(state.shape[-2], frames)["token_count"] > self.max_tokens:
            raise ValueError("Attention token budget exceeded; reduce chunk_seconds or crop_frames")
        tokens = self.frame_embed(aligned, aux_condition)
        positions = sinusoidal(torch.arange(frames, device=tokens.device), self.hidden_size)
        tokens = tokens + positions[None].to(tokens.dtype)
        embedding = self.time_embedder(time)
        for block in self.blocks:
            tokens = block(tokens, embedding)
        prediction = self.final_layer(tokens, embedding)
        return prediction.reshape(state.shape[0], frames, self.channels,
                                  self.frequency_bins).permute(0, 2, 3, 1).contiguous()
