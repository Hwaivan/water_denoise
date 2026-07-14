"""Tests for packed-complex primitives."""

import torch

from models.complex_layers import ComplexConv2d


def test_complex_conv_shape() -> None:
    layer = ComplexConv2d(2, 3, kernel_size=(3, 2), padding=(1, 0), bias=False)
    output = layer(torch.randn(4, 4, 17, 11))
    assert output.shape == (4, 6, 17, 10)


def test_complex_conv_matches_manual_multiplication() -> None:
    layer = ComplexConv2d(1, 1, kernel_size=(1, 1), bias=False)
    with torch.no_grad():
        layer.real_conv.weight.fill_(2.0)
        layer.imag_conv.weight.fill_(3.0)
    real = torch.tensor([[[[5.0]]]])
    imag = torch.tensor([[[[7.0]]]])
    output = layer(torch.cat((real, imag), dim=1))
    expected_real = 2.0 * real - 3.0 * imag
    expected_imag = 3.0 * real + 2.0 * imag
    assert torch.allclose(output, torch.cat((expected_real, expected_imag), dim=1))

