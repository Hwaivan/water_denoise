"""Length-aware scale-invariant signal-to-noise ratio loss."""

from typing import Dict, Optional

import torch
from torch import nn


def _valid_mask(reference: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
    if reference.ndim != 2:
        raise ValueError("Expected waveform shape [B,T]")
    if lengths is None:
        return torch.ones_like(reference)
    if lengths.ndim != 1 or lengths.shape[0] != reference.shape[0]:
        raise ValueError("lengths must have shape [B]")
    indices = torch.arange(reference.shape[1], device=reference.device)[None, :]
    return (indices < lengths[:, None]).to(reference.dtype)


def si_snr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Return per-example SI-SNR in dB for waveforms ``[B,T]``."""
    if estimate.shape != target.shape:
        raise ValueError("estimate and target must have identical shapes")
    mask = _valid_mask(target, lengths)
    counts = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    target_zm = (target - (target * mask).sum(1, keepdim=True) / counts) * mask
    estimate_zm = (estimate - (estimate * mask).sum(1, keepdim=True) / counts) * mask
    target_energy = target_zm.square().sum(1, keepdim=True)
    projection_scale = (estimate_zm * target_zm).sum(1, keepdim=True) / target_energy.clamp_min(eps)
    projection = projection_scale * target_zm
    residual = estimate_zm - projection
    projection_energy = projection.square().sum(1).clamp_min(eps)
    residual_energy = residual.square().sum(1)
    # A relative floor preserves SI-SNR's scale invariance even for exact
    # reconstruction, unlike adding the same absolute epsilon to both powers.
    ratio = projection_energy / (residual_energy + eps * projection_energy)
    values = 10.0 * torch.log10(ratio.clamp_min(eps))
    non_silent = target_energy.squeeze(1) > eps
    return torch.where(non_silent, values, torch.zeros_like(values))


# class SISNRLoss(nn.Module):
#     """Negative mean SI-SNR loss."""

#     def forward(
#         self, estimate: torch.Tensor, target: torch.Tensor, lengths: Optional[torch.Tensor] = None
#     ) -> torch.Tensor:
#         return -si_snr(estimate, target, lengths).mean()

# class SISNRLoss(nn.Module):
#     def forward(
#         self,
#         outputs: dict,
#         target: torch.Tensor,
#         lengths=None,
#     ) -> torch.Tensor:
#         estimate = outputs["waveform"]
#         return -si_snr(estimate, target, lengths).mean()

class SISNRLoss(nn.Module):
    """Negative mean SI-SNR loss."""

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return -si_snr(
            outputs["waveform"],
            target,
            lengths,
        ).mean()

