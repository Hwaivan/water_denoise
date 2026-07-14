"""Deep Complex Convolution Recurrent Network for speech enhancement."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as functional
from torch import nn

from utils.audio import STFTFrontend, channels_to_complex, complex_to_channels
from .complex_layers import (
    ComplexDecoderBlock,
    ComplexEncoderBlock,
    RecurrentBottleneck,
    complex_cat,
)
from .mask_reconstruction import build_reconstruction


def _align(inputs: torch.Tensor, frequency: int, frames: int) -> torch.Tensor:
    """Crop or right-pad decoder features to an exact skip-connection size."""
    inputs = inputs[:, :, :frequency, :frames]
    return functional.pad(
        inputs,
        (0, max(0, frames - inputs.shape[-1]), 0, max(0, frequency - inputs.shape[-2])),
    )


class DCCRN(nn.Module):
    """Configurable single-channel DCCRN.

    Input waveform is ``[B,T]``. The returned dictionary contains waveform
    ``[B,T]``, complex spectrum ``[B,F,N]``, and complex mask ``[B,F,N]``.
    """

    def __init__(
        self,
        stft_config: Dict[str, Any],
        encoder_channels: Sequence[int],
        kernel_size: Sequence[int] = (5, 2),
        stride: Sequence[int] = (2, 1),
        lstm_layers: int = 2,
        lstm_hidden_size: int = 256,
        reconstruction_mode: str = "dccrn_e",
        mask_activation: str = "tanh",
        use_complex_lstm: bool = False,
    ) -> None:
        super().__init__()
        if not encoder_channels:
            raise ValueError("encoder_channels must not be empty")
        if use_complex_lstm:
            raise NotImplementedError("Complex LSTM is reserved for a later extension")
        kernel = (int(kernel_size[0]), int(kernel_size[1]))
        strides = (int(stride[0]), int(stride[1]))
        padding = (kernel[0] // 2, 0)
        self.frontend = STFTFrontend(**stft_config)
        channels = [1] + [int(value) for value in encoder_channels]
        self.encoders = nn.ModuleList(
            [
                ComplexEncoderBlock(channels[i], channels[i + 1], kernel, strides, padding)
                for i in range(len(encoder_channels))
            ]
        )
        frequencies = self.frontend.n_fft // 2 + 1
        for _ in encoder_channels:
            frequencies = (frequencies + 2 * padding[0] - kernel[0]) // strides[0] + 1
        bottleneck_features = 2 * int(encoder_channels[-1]) * frequencies
        self.bottleneck = RecurrentBottleneck(
            bottleneck_features, int(lstm_hidden_size), int(lstm_layers)
        )
        decoder_outputs = list(reversed([int(value) for value in encoder_channels[:-1]])) + [1]
        current = int(encoder_channels[-1])
        self.decoders = nn.ModuleList()
        for index, output_channels in enumerate(decoder_outputs):
            skip_channels = int(encoder_channels[-1 - index])
            self.decoders.append(
                ComplexDecoderBlock(
                    current + skip_channels,
                    output_channels,
                    kernel,
                    strides,
                    padding,
                    activate=index != len(decoder_outputs) - 1,
                )
            )
            current = output_channels
        self.reconstruction = build_reconstruction(reconstruction_mode, mask_activation)

    def forward(
        self, mixture: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if mixture.ndim != 2:
            raise ValueError("DCCRN expects mixture shape [B,T]")
        original_length = mixture.shape[-1]
        noisy_spectrum = self.frontend.transform(mixture)
        features = complex_to_channels(noisy_spectrum)
        skips: List[torch.Tensor] = []
        for encoder in self.encoders:
            features = encoder(features)
            skips.append(features)
        features = self.bottleneck(features)
        targets: List[Tuple[int, int]] = [
            (skip.shape[-2], skip.shape[-1]) for skip in reversed(skips[:-1])
        ] + [(noisy_spectrum.shape[-2], noisy_spectrum.shape[-1])]
        for decoder, skip, target in zip(self.decoders, reversed(skips), targets):
            features = decoder(complex_cat(features, skip))
            features = _align(features, target[0], target[1])
        mask = channels_to_complex(features).squeeze(1)
        estimated_spectrum = self.reconstruction(noisy_spectrum, mask)
        waveform = self.frontend.inverse(estimated_spectrum, length=original_length)
        waveform = torch.nan_to_num(waveform)
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] != mixture.shape[0]:
                raise ValueError("lengths must have shape [B]")
            valid = torch.arange(original_length, device=mixture.device)[None, :] < lengths[:, None]
            waveform = waveform * valid.to(waveform.dtype)
        return {"waveform": waveform, "spectrum": estimated_spectrum, "mask": mask}


def build_model(config: Dict[str, Any]) -> DCCRN:
    """Build DCCRN from top-level experiment configuration."""
    model_config = dict(config["model"])
    model_config.pop("name", None)
    return DCCRN(stft_config=dict(config["stft"]), **model_config)

