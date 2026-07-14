"""Complex neural-network layers using packed real/imaginary channels."""

from typing import Optional, Tuple

import torch
from torch import nn


def split_complex(inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split packed ``[B,2C,F,T]`` features into real and imaginary parts."""
    if inputs.ndim != 4 or inputs.shape[1] % 2:
        raise ValueError("Expected packed complex features [B,2C,F,T]")
    return inputs.chunk(2, dim=1)


def complex_cat(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Concatenate two packed complex tensors along their complex channels."""
    first_real, first_imag = split_complex(first)
    second_real, second_imag = split_complex(second)
    return torch.cat((first_real, second_real, first_imag, second_imag), dim=1)


class ComplexConv2d(nn.Module):
    """Complex 2-D convolution.

    Input: ``[B,2*in_channels,F,T]``. Output:
    ``[B,2*out_channels,F_out,T_out]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.real_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.imag_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.real_bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.imag_bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        real, imag = split_complex(inputs)
        output_real = self.real_conv(real) - self.imag_conv(imag)
        output_imag = self.imag_conv(real) + self.real_conv(imag)
        if self.real_bias is not None:
            output_real = output_real + self.real_bias.view(1, -1, 1, 1)
            output_imag = output_imag + self.imag_bias.view(1, -1, 1, 1)
        return torch.cat((output_real, output_imag), dim=1)


class ComplexConvTranspose2d(nn.Module):
    """Complex transposed 2-D convolution on packed complex features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        output_padding: Tuple[int, int] = (0, 0),
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.real_conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            bias=False,
        )
        self.imag_conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            bias=False,
        )
        self.real_bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.imag_bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        real, imag = split_complex(inputs)
        output_real = self.real_conv(real) - self.imag_conv(imag)
        output_imag = self.imag_conv(real) + self.real_conv(imag)
        if self.real_bias is not None:
            output_real = output_real + self.real_bias.view(1, -1, 1, 1)
            output_imag = output_imag + self.imag_bias.view(1, -1, 1, 1)
        return torch.cat((output_real, output_imag), dim=1)


class ComplexBatchNorm2d(nn.Module):
    """Stable component-wise complex batch normalization.

    This implementation normalizes real and imaginary components independently;
    it deliberately avoids a fragile per-batch covariance inverse.
    """

    def __init__(self, channels: int, eps: float = 1.0e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.real_norm = nn.BatchNorm2d(channels, eps=eps, momentum=momentum)
        self.imag_norm = nn.BatchNorm2d(channels, eps=eps, momentum=momentum)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        real, imag = split_complex(inputs)
        return torch.cat((self.real_norm(real), self.imag_norm(imag)), dim=1)


class ComplexEncoderBlock(nn.Module):
    """Complex convolution, normalization, and PReLU encoder block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
    ) -> None:
        super().__init__()
        self.conv = ComplexConv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = ComplexBatchNorm2d(out_channels)
        self.activation = nn.PReLU(2 * out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(inputs)))


class ComplexDecoderBlock(nn.Module):
    """Complex transposed-convolution decoder block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        activate: bool = True,
    ) -> None:
        super().__init__()
        self.conv = ComplexConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.norm = ComplexBatchNorm2d(out_channels) if activate else nn.Identity()
        self.activation = nn.PReLU(2 * out_channels) if activate else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(inputs)))


class RecurrentBottleneck(nn.Module):
    """Real-valued LSTM bottleneck operating on flattened complex features."""

    def __init__(self, feature_size: int, hidden_size: int, layers: int) -> None:
        super().__init__()
        self.feature_size = int(feature_size)
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
        )
        self.projection = nn.Linear(hidden_size, feature_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # [B,2C,F,N] -> [B,N,2CF]
        batch, channels, frequencies, frames = inputs.shape
        sequence = inputs.permute(0, 3, 1, 2).contiguous().view(batch, frames, -1)
        if sequence.shape[-1] != self.feature_size:
            raise ValueError("Unexpected bottleneck feature size")
        sequence, _ = self.lstm(sequence)
        sequence = self.projection(sequence)
        return sequence.view(batch, frames, channels, frequencies).permute(0, 2, 3, 1)

