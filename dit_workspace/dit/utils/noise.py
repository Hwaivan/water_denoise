"""Centralized stochastic scale convention for all processes and samplers."""
import math
import torch


def sample_noise(reference, convention, generator=None):
    if convention not in ("circular_complex_unit_energy", "real_unit_variance"):
        raise ValueError("Unsupported noise convention")
    if convention == "circular_complex_unit_energy":
        if reference.ndim != 4 or reference.shape[1] != 2:
            raise ValueError("Circular noise requires real/imag channels")
        # Keep the legacy real draw followed by imaginary draw (including B>1).
        shape = reference[:, 0].shape
        real = torch.randn(shape, device=reference.device, dtype=reference.dtype, generator=generator)
        imag = torch.randn(shape, device=reference.device, dtype=reference.dtype, generator=generator)
        return torch.stack([real, imag], 1) / math.sqrt(2)
    return torch.randn(reference.shape, device=reference.device, dtype=reference.dtype, generator=generator)
