"""Pluggable DCCRN complex-mask reconstruction strategies."""

from typing import Callable, Dict

import torch
from torch import nn


class MaskReconstruction(nn.Module):
    """Base class mapping noisy spectrum and complex mask to an estimate."""

    def forward(self, noisy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DCCRNEReconstruction(MaskReconstruction):
    """Magnitude/phase reconstruction used by DCCRN-E."""

    def __init__(self, activation: str = "tanh", eps: float = 1.0e-8) -> None:
        super().__init__()
        if activation not in ("tanh", "sigmoid"):
            raise ValueError("mask_activation must be tanh or sigmoid")
        self.activation = activation
        self.eps = float(eps)

    def forward(self, noisy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        magnitude = torch.sqrt(mask.real.square() + mask.imag.square() + self.eps)
        magnitude = torch.tanh(magnitude) if self.activation == "tanh" else torch.sigmoid(magnitude)
        phase = torch.atan2(mask.imag, mask.real)
        estimated_magnitude = noisy.abs() * magnitude
        estimated_phase = torch.angle(noisy) + phase
        return torch.polar(estimated_magnitude, estimated_phase)


class DCCRNCReconstruction(MaskReconstruction):
    """Standard complex multiplication (DCCRN-C/cIRM)."""

    def forward(self, noisy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bounded = torch.complex(torch.tanh(mask.real), torch.tanh(mask.imag))
        return noisy * bounded


class DCCRNReconstruction(MaskReconstruction):
    """Independent real/imaginary masking (DCCRN-R)."""

    def forward(self, noisy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.complex(noisy.real * torch.tanh(mask.real), noisy.imag * torch.tanh(mask.imag))


_RECONSTRUCTION: Dict[str, Callable[..., MaskReconstruction]] = {
    "dccrn_e": DCCRNEReconstruction,
    "dccrn_c": DCCRNCReconstruction,
    "dccrn_r": DCCRNReconstruction,
}


def build_reconstruction(mode: str, mask_activation: str = "tanh") -> MaskReconstruction:
    """Build a registered reconstruction strategy."""
    key = mode.lower()
    if key not in _RECONSTRUCTION:
        raise ValueError("Unknown reconstruction mode: {}".format(mode))
    if key == "dccrn_e":
        return _RECONSTRUCTION[key](activation=mask_activation)
    return _RECONSTRUCTION[key]()

