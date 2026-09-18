"""CDiffuSE conditional forward process and paper regression target."""
import torch
from .schedule import CDiffuSESchedule


def _wave(x):
    if x.ndim == 2: return x.unsqueeze(1)
    if x.ndim == 3 and x.shape[1] == 1: return x
    raise ValueError("waveform must be [B,L] or [B,1,L]")


def q_sample(schedule: CDiffuSESchedule, clean, noisy, t, epsilon=None):
    """Return ``(xt, epsilon, mean)``; all waveform tensors are real ``[B,1,L]``."""
    clean, noisy = _wave(clean), _wave(noisy)
    if clean.shape != noisy.shape or not torch.isfinite(clean).all() or not torch.isfinite(noisy).all():
        raise ValueError("clean/noisy must be aligned and finite")
    epsilon = torch.randn_like(clean) if epsilon is None else _wave(epsilon)
    ab = schedule.extract(schedule.alpha_bar, t, clean)
    m = schedule.extract(schedule.m, t, clean)
    delta = schedule.extract(schedule.delta, t, clean)
    mean = (1-m)*torch.sqrt(ab)*clean + m*torch.sqrt(ab)*noisy
    xt = mean + torch.sqrt(delta.clamp_min(schedule.variance_floor))*epsilon
    return xt, epsilon, mean


def combined_noise_target(schedule, clean, noisy, epsilon, t):
    """Paper target: [m sqrt(ab)(y-x0)+sqrt(delta) eps]/sqrt(1-ab)."""
    clean, noisy, epsilon = _wave(clean), _wave(noisy), _wave(epsilon)
    ab = schedule.extract(schedule.alpha_bar, t, clean)
    m = schedule.extract(schedule.m, t, clean)
    delta = schedule.extract(schedule.delta, t, clean)
    return (m*torch.sqrt(ab)*(noisy-clean) + torch.sqrt(delta.clamp_min(schedule.variance_floor))*epsilon) / torch.sqrt((1-ab).clamp_min(schedule.variance_floor))
