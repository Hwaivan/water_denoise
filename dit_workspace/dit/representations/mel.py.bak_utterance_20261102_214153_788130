"""Fixed Slaney log-Mel state/condition codec with reproducible statistics.

The inverse is deliberately a noisy-STFT gain reconstruction, NOT a
mathematically nonexistent inverse Mel transform or a speech vocoder.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn

from .stft import ComplexSTFTTransform


def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    # Slaney scale (librosa/torchaudio's mel_scale='slaney').
    linear = hz / (200.0 / 3.0)
    log_mel = 15.0 + torch.log(hz.clamp_min(1.0) / 1000.0) / (math.log(6.4) / 27.0)
    return torch.where(hz >= 1000.0, log_mel, linear)


def _mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
    linear = (200.0 / 3.0) * mels
    log_hz = 1000.0 * torch.exp((mels - 15.0) * (math.log(6.4) / 27.0))
    return torch.where(mels >= 15.0, log_hz, linear)


def slaney_mel_bank(sample_rate: int, n_fft: int, n_mels: int,
                    f_min: float, f_max: float, norm: str = "slaney") -> torch.Tensor:
    if n_mels < 1 or not 0 <= f_min < f_max <= sample_rate / 2:
        raise ValueError("Require n_mels>=1 and 0<=f_min<f_max<=Nyquist")
    if norm not in ("slaney", "none"):
        raise ValueError("Mel filter normalization must be 'slaney' or 'none'")
    fft_f = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1, dtype=torch.float64)
    borders_mel = torch.linspace(_hz_to_mel(torch.tensor(f_min, dtype=torch.float64)),
                                 _hz_to_mel(torch.tensor(f_max, dtype=torch.float64)),
                                 n_mels + 2)
    hz = _mel_to_hz(borders_mel)
    up = (fft_f[None] - hz[:-2, None]) / (hz[1:-1, None] - hz[:-2, None])
    down = (hz[2:, None] - fft_f[None]) / (hz[2:, None] - hz[1:-1, None])
    weights = torch.minimum(up, down).clamp_min(0.0)
    if norm == "slaney":
        weights *= 2.0 / (hz[2:] - hz[:-2])[:, None]
    if (weights.sum(dim=1) == 0).any():
        raise ValueError("Some Mel bands are empty; increase n_fft or reduce n_mels")
    return weights.float()  # [n_mels, n_fft//2+1]


class LogMelRepresentation(ComplexSTFTTransform):
    """Encode [B,S] -> [B,1,M,T]; decode using observed complex STFT.

    Normalization uses fixed per-band (mean, std), estimated from TRAIN data
    only. Clean and noisy MUST use the same statistics.
    """

    def __init__(self, mel: dict, reconstruction: dict | None = None,
                 sample_rate: int = 16000, **stft_and_compression):
        super().__init__(**stft_and_compression)
        cfg = dict(mel)
        self.sample_rate = int(sample_rate)
        self.n_mels = int(cfg.get("n_mels", 128))
        self.channels = 1
        self.bins = self.n_mels
        self.type = "log_mel"
        self.power = float(cfg.get("power", 2.0))
        self.log_eps = float(cfg.get("log_eps", 1e-6))
        self.normalization = str(cfg.get("normalization", "dataset"))
        self.stats_path = cfg.get("stats_path")
        self.reconstruction = dict(reconstruction or {"type": "mel_gain_noisy_phase"})
        if not self.onesided:
            raise ValueError("Mel representation requires onesided=True")
        if self.power <= 0 or self.log_eps <= 0:
            raise ValueError("Mel power and epsilon must be positive")
        if self.normalization not in ("dataset", "none"):
            raise ValueError("Mel normalization must be dataset or none")
        if self.reconstruction.get("type", "mel_gain_noisy_phase") != "mel_gain_noisy_phase":
            raise NotImplementedError("Only mel_gain_noisy_phase reconstruction implemented")
        self.gain_min = float(self.reconstruction.get("gain_min", 0.05))
        self.gain_max = float(self.reconstruction.get("gain_max", 4.0))
        if not 0 < self.gain_min <= 1 <= self.gain_max:
            raise ValueError("Require 0 < gain_min <= 1 <= gain_max")
        f_max = cfg.get("f_max")
        if f_max is None:
            f_max = self.sample_rate / 2.0
        self.register_buffer("mel_weights", slaney_mel_bank(
            self.sample_rate, self.n_fft, self.n_mels,
            float(cfg.get("f_min", 0.0)), float(f_max),
            str(cfg.get("norm", "slaney"))), persistent=False)
        self.register_buffer("mean", torch.zeros(self.n_mels), persistent=False)
        self.register_buffer("std", torch.ones(self.n_mels), persistent=False)
        if self.normalization == "dataset":
            if not self.stats_path:
                raise ValueError("log_mel dataset normalization requires stats_path")
            path = Path(self.stats_path)
            if not path.is_file():
                raise FileNotFoundError(f"Mel stats missing: {path}")
            expected = cfg.get("stats_sha256")
            if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Mel stats SHA256 differs from training configuration")
            obj = torch.load(path, map_location="cpu", weights_only=True)
            mean = obj["mean"].float().reshape(-1)
            std = obj["std"].float().reshape(-1)
            if mean.numel() != self.n_mels or std.numel() != self.n_mels:
                raise ValueError("Mel stats band dimension mismatch")
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
                raise ValueError("Mel stats must be finite and std strictly positive")
            self.mean.copy_(mean)
            self.std.copy_(std)

    def log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return unnormalized log-Mel power [B,M,T] for statistics or modeling."""
        spectrum = self.stft(waveform)
        powers = spectrum.abs().pow(self.power)
        mel = torch.einsum("mf,bft->bmt", self.mel_weights.to(powers), powers)
        return torch.log(mel.clamp_min(self.log_eps))

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        value = self.log_mel(waveform)
        value = (value - self.mean[None, :, None]) / self.std[None, :, None]
        return value[:, None]

    def decode(self, representation: torch.Tensor, length: int,
               noisy_phase=None, noisy_waveform: torch.Tensor | None = None) -> torch.Tensor:
        if representation.ndim != 4 or representation.shape[1:3] != (1, self.n_mels):
            raise ValueError("log_mel estimate must have shape [B,1,M,T]")
        if noisy_waveform is None:
            raise ValueError("Mel gain decoder requires original noisy waveform")
        noisy_spectrum = self.stft(noisy_waveform.float())
        if noisy_spectrum.shape[-1] != representation.shape[-1]:
            raise ValueError("noisy/estimated Mel frame counts differ")
        noisy_log = self.log_mel(noisy_waveform.float())
        target_log = representation[:, 0].float() * self.std[None, :, None] + self.mean[None, :, None]
        # A Mel power ratio, converted to a magnitude gain; no inverse Mel claim.
        log_gain = ((target_log - noisy_log) / self.power).clamp(
            min=math.log(self.gain_min), max=math.log(self.gain_max))
        mel_gain = log_gain.exp()
        w = self.mel_weights.to(mel_gain)
        column_sum = w.sum(0)  # mel filters may not cover DC or Nyquist
        freq_gain = torch.einsum("fm,bmt->bft", w.T, mel_gain) / column_sum.clamp_min(1e-12)[None, :, None]
        freq_gain = torch.where(column_sum[None, :, None] > 0, freq_gain, torch.ones_like(freq_gain))
        return self.istft(noisy_spectrum * freq_gain, length=length)
