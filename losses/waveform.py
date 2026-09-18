from typing import Optional

import torch
from torch import nn


def valid_mask(
    reference: torch.Tensor,
    lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    if lengths is None:
        return torch.ones_like(reference)

    indices = torch.arange(
        reference.shape[-1],
        device=reference.device,
    )[None, :]

    return (indices < lengths[:, None]).to(reference.dtype)


class WaveformL1Loss(nn.Module):
    def forward(
        self,
        outputs: dict,
        target: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        estimate = outputs["waveform"]

        mask = valid_mask(target, lengths)
        error = torch.abs(estimate - target) * mask

        counts = mask.sum(dim=1).clamp_min(1.0)
        return (error.sum(dim=1) / counts).mean()


class WaveformMSELoss(nn.Module):
    def forward(
        self,
        outputs: dict,
        target: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        estimate = outputs["waveform"]

        mask = valid_mask(target, lengths)
        error = (estimate - target).square() * mask

        counts = mask.sum(dim=1).clamp_min(1.0)
        return (error.sum(dim=1) / counts).mean()