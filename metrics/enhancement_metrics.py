"""Dependency-free waveform metrics for enhancement evaluation."""

from typing import Dict, Optional

import torch

from losses.si_snr import si_snr


def _masked_values(
    estimate: torch.Tensor, target: torch.Tensor, lengths: Optional[torch.Tensor], eps: float
) -> Dict[str, torch.Tensor]:
    if lengths is None:
        mask = torch.ones_like(target)
    else:
        mask = (
            torch.arange(target.shape[1], device=target.device)[None, :] < lengths[:, None]
        ).to(target.dtype)
    count = mask.sum(1).clamp_min(1.0)
    error = (estimate - target) * mask
    target_energy = (target.square() * mask).sum(1)
    error_energy = error.square().sum(1)
    sdr = 10.0 * torch.log10((target_energy + eps) / (error_energy + eps))
    l1 = error.abs().sum(1) / count
    estimate_mean = (estimate * mask).sum(1, keepdim=True) / count[:, None]
    target_mean = (target * mask).sum(1, keepdim=True) / count[:, None]
    estimate_zm = (estimate - estimate_mean) * mask
    target_zm = (target - target_mean) * mask
    corr = (estimate_zm * target_zm).sum(1) / torch.sqrt(
        estimate_zm.square().sum(1) * target_zm.square().sum(1) + eps
    )
    return {"sdr": sdr, "l1": l1, "corr": corr}


def compute_metrics(
    estimate: torch.Tensor,
    target: torch.Tensor,
    mixture: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    eps: float = 1.0e-8,
) -> Dict[str, torch.Tensor]:
    """Compute per-example SI-SNR, simple SDR, improvements, L1, correlation.

    ``sdr`` is signal-to-error SDR and is intentionally not called BSS Eval SDR.
    """
    enhanced = _masked_values(estimate, target, lengths, eps)
    baseline = _masked_values(mixture, target, lengths, eps)
    output_si_snr = si_snr(estimate, target, lengths, eps)
    input_si_snr = si_snr(mixture, target, lengths, eps)
    return {
        "si_snr": output_si_snr,
        "si_snri": output_si_snr - input_si_snr,
        "sdr": enhanced["sdr"],
        "sdri": enhanced["sdr"] - baseline["sdr"],
        "l1": enhanced["l1"],
        "corr": enhanced["corr"],
    }

