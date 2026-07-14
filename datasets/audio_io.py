"""Audio I/O with torchaudio-first and soundfile fallback backends."""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as functional


def _mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        return waveform.float()
    if waveform.ndim == 2:
        return waveform.float().mean(dim=0)
    raise ValueError("Audio must have one sample and one optional channel dimension")


def resample_audio(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    """Resample one-dimensional audio, preferring torchaudio."""
    if source_rate == target_rate:
        return waveform
    try:
        import torchaudio

        return torchaudio.functional.resample(waveform, source_rate, target_rate)
    except (ImportError, OSError):
        output_length = max(1, int(round(waveform.numel() * target_rate / source_rate)))
        return functional.interpolate(
            waveform.view(1, 1, -1), output_length, mode="linear", align_corners=False
        ).view(-1)


def load_audio(path: str, sample_rate: int) -> Tuple[torch.Tensor, int]:
    """Load, downmix, and resample an audio file."""
    audio_path = Path(path).expanduser()
    if not audio_path.is_file():
        raise FileNotFoundError("Audio file not found: {}".format(audio_path))
    try:
        import torchaudio

        waveform, source_rate = torchaudio.load(str(audio_path))
        waveform = _mono(waveform)
    except (ImportError, OSError):
        try:
            import soundfile as soundfile
        except ImportError as error:
            raise RuntimeError("Install torchaudio or soundfile to read audio") from error
        samples, source_rate = soundfile.read(str(audio_path), always_2d=True)
        waveform = torch.from_numpy(np.asarray(samples, dtype=np.float32).mean(axis=1))
    waveform = resample_audio(waveform, int(source_rate), int(sample_rate))
    return waveform.contiguous(), int(sample_rate)


def save_audio(path: str, waveform: torch.Tensor, sample_rate: int) -> None:
    """Save mono waveform using torchaudio or soundfile."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    samples = waveform.detach().float().cpu().clamp(-1.0, 1.0)
    try:
        import torchaudio

        torchaudio.save(str(output), samples.view(1, -1), sample_rate)
    except (ImportError, OSError):
        try:
            import soundfile as soundfile
        except ImportError as error:
            raise RuntimeError("Install torchaudio or soundfile to save audio") from error
        soundfile.write(str(output), samples.numpy(), sample_rate)

