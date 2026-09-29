"""Real model channels with reversible complex-STFT boundary adapters."""
import torch
from .stft import ComplexSTFTTransform, complex_to_channels, channels_to_complex


class SpectralRepresentation(ComplexSTFTTransform):
    def __init__(self, type="complex_ri", phase_source="noisy", **kwargs):
        super().__init__(**kwargs)
        if type not in ("complex_ri", "magnitude"):
            raise ValueError("Unknown spectral representation")
        if phase_source != "noisy":
            raise ValueError("Only noisy phase reconstruction is implemented")
        self.type, self.phase_source = type, phase_source
        self.channels = 2 if type == "complex_ri" else 1

    def encode(self, waveform):
        spectrum = self.compress(self.stft(waveform))
        return complex_to_channels(spectrum) if self.channels == 2 else spectrum.abs()[:, None]

    def to_spectrum(self, representation, noisy_phase=None):
        if representation.ndim != 4 or representation.shape[1] != self.channels:
            raise ValueError("Expected [B,C,F,T] representation")
        if self.channels == 2:
            compressed = channels_to_complex(representation.float())
        else:
            if noisy_phase is None or noisy_phase.shape != representation[:, 0].shape:
                raise ValueError("Magnitude reconstruction requires aligned noisy phase")
            # This is the ONLY magnitude clamp: generative states remain unconstrained.
            compressed = torch.polar(representation[:, 0].float().clamp_min(0), noisy_phase)
        return self.decompress(compressed)

    def decode(self, representation, length, noisy_phase=None):
        return self.istft(self.to_spectrum(representation, noisy_phase), length)
