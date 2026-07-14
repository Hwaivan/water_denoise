"""Numerical and invariance tests for SI-SNR."""

import torch

from losses.si_snr import SISNRLoss, si_snr


def test_si_snr_is_scale_invariant() -> None:
    target = torch.randn(4, 1000)
    original = si_snr(target, target)
    scaled = si_snr(2.0 * target, target)
    assert torch.allclose(original, scaled, atol=0.2, rtol=0.0)


def test_matching_signal_beats_random_noise() -> None:
    target = torch.randn(4, 1000)
    matching = si_snr(target, target).mean()
    random = si_snr(torch.randn_like(target), target).mean()
    assert matching > random + 30.0
    assert SISNRLoss()(target, target) < -50.0


def test_length_mask_ignores_padding() -> None:
    target = torch.randn(2, 100)
    estimate = target.clone()
    estimate[1, 60:] = 1000.0
    lengths = torch.tensor([100, 60])
    masked = si_snr(estimate, target, lengths)
    baseline = si_snr(target, target, lengths)
    assert torch.allclose(masked, baseline)


def test_silence_is_finite() -> None:
    values = si_snr(torch.zeros(2, 100), torch.zeros(2, 100))
    assert torch.isfinite(values).all()

