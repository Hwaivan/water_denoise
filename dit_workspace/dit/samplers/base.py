"""Shared waveform decoding and bounded long-audio overlap-add for every process."""
import time
from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass
class SampleResult:
    representation: torch.Tensor
    waveform: torch.Tensor
    nfe: int
    inference_time: float
    rtf: float

    @property
    def spectrum(self):
        """Compatibility alias: compressed representation, not a physical STFT."""
        return self.representation


class WaveformSampler:
    def __init__(self, representation, solver):
        self.representation, self.solver = representation, solver

    def sample_spectrum(self, model, condition, generator=None):
        return self.solver.sample(model, condition, generator)

    @torch.inference_mode()
    def sample_waveform(self, model, noisy, lengths=None, sample_rate=16000, generator=None):
        if noisy.ndim != 2 or noisy.shape[-1] == 0:
            raise ValueError("Expected nonempty [B,L] audio")
        if noisy.device.type == "cuda":
            torch.cuda.synchronize(noisy.device)
        started = time.perf_counter()
        self.representation.to(noisy.device)
        self.solver.process.to(noisy.device)
        # Reflect STFT cannot pad an input shorter than n_fft/2.
        original_length = noisy.shape[-1]
        noisy = F.pad(noisy, (0, max(0, self.representation.n_fft // 2 + 1-original_length)))
        with torch.autocast(device_type=noisy.device.type, enabled=False):
            condition = self.representation.encode(noisy.float())
            phase = self.representation.stft(noisy.float()).angle() if self.representation.channels == 1 else None
            estimate, nfe = self.sample_spectrum(model, condition, generator)
            waveform = self.representation.decode(estimate, noisy.shape[-1], phase)[..., :original_length]
        if lengths is not None:
            waveform *= (torch.arange(original_length, device=noisy.device)[None] < lengths[:, None])
        if not torch.isfinite(waveform).all():
            raise FloatingPointError("Non-finite decoded waveform")
        if noisy.device.type == "cuda":
            torch.cuda.synchronize(noisy.device)
        elapsed = time.perf_counter()-started
        duration = (lengths.sum().item() if lengths is not None else waveform.numel()) / sample_rate
        return SampleResult(estimate, waveform, nfe, elapsed, elapsed/max(duration, 1e-12))


@torch.inference_mode()
def enhance_long_waveform(sampler, model, waveform, sample_rate, generator=None,
                          chunk_samples=None, overlap_samples=0, fallback_on_oom=True):
    if waveform.ndim != 1 or waveform.numel() == 0:
        raise ValueError("Expected nonempty mono waveform")
    if chunk_samples is None:
        chunk_samples = sample_rate  # Safe default for global attention.
        overlap_samples = min(overlap_samples, chunk_samples // 2)
    if chunk_samples < 1 or not 0 <= overlap_samples < chunk_samples:
        raise ValueError("Invalid chunk or overlap length")
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        if waveform.numel() <= chunk_samples:
            return sampler.sample_waveform(model, waveform[None].to(device),
                torch.tensor([waveform.numel()], device=device), sample_rate, generator)
        output = torch.zeros(waveform.numel(), device=device)
        weights = torch.zeros_like(output)
        nfe, last = 0, None
        for start in range(0, waveform.numel(), chunk_samples-overlap_samples):
            valid = min(chunk_samples, waveform.numel()-start)
            chunk = F.pad(waveform[start:start+valid], (0, chunk_samples-valid))
            last = sampler.sample_waveform(model, chunk[None].to(device),
                torch.tensor([valid], device=device), sample_rate, generator)
            window = torch.ones(valid, device=device)
            fade = min(overlap_samples, valid)
            if fade:
                # Strictly positive ramps avoid uncovered samples at tiny overlaps.
                ramp = torch.linspace(0, 1, fade+2, device=device)[1:-1]
                if start:
                    window[:fade] *= ramp
                if start+valid < waveform.numel():
                    window[-fade:] *= ramp.flip(0)
            output[start:start+valid] += last.waveform[0,:valid] * window
            weights[start:start+valid] += window
            nfe += last.nfe
            if start+valid == waveform.numel():
                break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter()-started
        return SampleResult(last.representation, (output/weights)[None], nfe, elapsed,
                            elapsed/(waveform.numel()/sample_rate))
    except torch.cuda.OutOfMemoryError:
        if not fallback_on_oom or chunk_samples <= sampler.representation.n_fft * 2:
            raise
        torch.cuda.empty_cache()
        result = enhance_long_waveform(sampler, model, waveform, sample_rate, generator,
            chunk_samples//2, min(overlap_samples, chunk_samples//4), True)
        result.inference_time = time.perf_counter() - started
        result.rtf = result.inference_time / (waveform.numel()/sample_rate)
        return result


def enhance_configured(sampler, model, waveform, config, generator):
    sr = int(config["data"]["sample_rate"])
    cfg = config["inference"]
    return enhance_long_waveform(sampler, model, waveform, sr, generator,
        round(float(cfg["chunk_seconds"]) * sr), round(float(cfg["overlap_seconds"]) * sr),
        bool(cfg.get("fallback_on_oom", True)))
