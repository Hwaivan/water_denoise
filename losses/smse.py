"""Magnitude-spectrum mean squared error loss."""

from typing import Any, Dict, Optional

import torch
from torch import nn

from utils.audio import STFTFrontend


class SMSELoss(nn.Module):
    """纯净信号与增强信号幅度谱的时频点均方误差。"""

    def __init__(self, stft_config: Dict[str, Any]) -> None:
        super().__init__()
        self.frontend = STFTFrontend(**stft_config)
        self.n_fft = int(stft_config["n_fft"])
        self.hop_length = int(stft_config["hop_length"])
        self.center = bool(stft_config.get("center", True))

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        enhanced_spectrum = outputs["spectrum"]
        target_spectrum = self.frontend.transform(target)

        if enhanced_spectrum.shape != target_spectrum.shape:
            raise ValueError(
                "Enhanced and target spectra must have identical shapes"
            )

        squared_error = (
            enhanced_spectrum.abs() - target_spectrum.abs()
        ).square()

        if lengths is None:
            return squared_error.mean()

        frame_count = squared_error.shape[-1]

        if self.center:
            valid_frame_counts = (
                torch.div(
                    lengths,
                    self.hop_length,
                    rounding_mode="floor",
                )
                + 1
            )
        else:
            valid_frame_counts = (
                torch.div(
                    (lengths - self.n_fft).clamp_min(0),
                    self.hop_length,
                    rounding_mode="floor",
                )
                + 1
            )

        valid_frame_counts = valid_frame_counts.clamp(
            min=1,
            max=frame_count,
        )

        frame_indices = torch.arange(
            frame_count,
            device=squared_error.device,
        )[None, :]

        frame_mask = (
            frame_indices < valid_frame_counts[:, None]
        ).to(squared_error.dtype)

        frame_mask = frame_mask[:, None, :]

        error_sum = (
            squared_error * frame_mask
        ).sum(dim=(1, 2))

        valid_points = (
            valid_frame_counts.to(squared_error.dtype)
            * squared_error.shape[1]
        ).clamp_min(1.0)

        return (error_sum / valid_points).mean()