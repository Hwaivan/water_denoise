"""STFT frontend and canonical complex-tensor conversions."""

from typing import Optional

import torch
from torch import nn


def complex_to_channels(spectrum: torch.Tensor) -> torch.Tensor:
    """Convert complex ``[B,C,F,N]`` or ``[B,F,N]`` to packed real channels.

    The packed layout is ``[real channels, imaginary channels]``. A spectrum
    without a channel dimension is treated as one complex channel.
    """
    if not torch.is_complex(spectrum):
        raise TypeError("Expected a native complex tensor")
    if spectrum.ndim == 3:
        spectrum = spectrum.unsqueeze(1)
    if spectrum.ndim != 4:
        raise ValueError("Expected spectrum shape [B,F,N] or [B,C,F,N]")
    return torch.cat((spectrum.real, spectrum.imag), dim=1)


def channels_to_complex(channels: torch.Tensor) -> torch.Tensor:
    """Convert packed real ``[B,2C,F,N]`` into complex ``[B,C,F,N]``."""
    if channels.ndim != 4 or channels.shape[1] % 2:
        raise ValueError("Expected packed channels with shape [B,2C,F,N]")
    real, imag = channels.chunk(2, dim=1)
    return torch.complex(real, imag)


class STFTFrontend(nn.Module):
    """Reusable waveform/STFT transform.

    Input waveform shape is ``[B,T]`` and output spectrum shape is
    ``[B,F,Frames]``. The registered window follows the module device.
    """

    def __init__(
        self,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 100,
        window: str = "hann",
        center: bool = True,
    ) -> None:
        super().__init__()
        if window != "hann":
            raise ValueError("Only the hann window is currently supported")
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.center = bool(center)
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

    def transform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Transform float waveform ``[B,T]`` to complex STFT ``[B,F,N]``."""
        if waveform.ndim != 2:
            raise ValueError("Expected waveform shape [B,T]")
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(dtype=waveform.dtype),
            center=self.center,
            return_complex=True,
        )

    def inverse(self, spectrum: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """Transform complex STFT ``[B,F,N]`` back to waveform ``[B,T]``."""
        if spectrum.ndim != 3 or not torch.is_complex(spectrum):
            raise ValueError("Expected native complex spectrum shape [B,F,N]")
        return torch.istft(
            spectrum,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(dtype=spectrum.real.dtype),
            center=self.center,
            length=length,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.transform(waveform)

