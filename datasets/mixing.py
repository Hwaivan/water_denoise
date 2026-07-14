"""Numerically safe waveform preparation and SNR-controlled mixing."""

from typing import Optional, Tuple

import torch


def remove_dc(waveform: torch.Tensor) -> torch.Tensor:
    """Remove a waveform's DC component."""
    return waveform - waveform.mean()


def crop_or_pad(
    waveform: torch.Tensor,
    length: int,
    generator: Optional[torch.Generator] = None,
    random_crop: bool = True,
) -> torch.Tensor:
    """Crop or right-pad a one-dimensional waveform to an exact length."""
    if waveform.ndim != 1:
        raise ValueError("Expected a one-dimensional waveform")
    if waveform.numel() < length:
        repeats = (length + waveform.numel() - 1) // max(1, waveform.numel())
        if waveform.numel() == 0:
            return torch.zeros(length, dtype=torch.float32)
        waveform = waveform.repeat(repeats)
    max_start = waveform.numel() - length
    if random_crop and max_start > 0:
        start = int(torch.randint(max_start + 1, (1,), generator=generator).item())
    else:
        start = 0
    return waveform[start : start + length]


def mix_at_snr(
    target: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
    eps: float = 1.0e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scale noise to the requested signal-to-noise ratio and mix.

    Returns ``(mixture, scaled_noise)`` with the same shape as the inputs.
    """
    if target.shape != noise.shape:
        raise ValueError("Target and noise must have identical shapes")
    target = remove_dc(target.float())
    noise = remove_dc(noise.float())
    target_energy = target.square().mean()
    noise_energy = noise.square().mean()
    if target_energy <= eps:
        target = torch.zeros_like(target)
        target_energy = target.new_tensor(eps)
    if noise_energy <= eps:
        noise = torch.randn(noise.shape, generator=None, device=noise.device, dtype=noise.dtype)
        noise = remove_dc(noise)
        noise_energy = noise.square().mean().clamp_min(eps)
    ratio = target.new_tensor(10.0).pow(float(snr_db) / 10.0)
    scale = torch.sqrt(target_energy / (noise_energy * ratio + eps))
    scaled_noise = noise * scale
    mixture = torch.nan_to_num(target + scaled_noise)
    return mixture, scaled_noise

