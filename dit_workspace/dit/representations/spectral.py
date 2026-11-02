"""Real model channels with reversible complex-STFT boundary adapters."""
import torch
from .stft import ComplexSTFTTransform, complex_to_channels, channels_to_complex


class SpectralRepresentation(ComplexSTFTTransform):
    def __init__(self, type="complex_ri", phase_source="noisy", normalization="none", utterance_std_floor=1e-3, **kwargs):
        super().__init__(**kwargs)
        if type not in ("complex_ri", "magnitude", "log_magnitude"):
            raise ValueError("Unknown spectral representation")
        if phase_source != "noisy":
            raise ValueError("Only noisy phase reconstruction is implemented")
        self.type, self.phase_source = type, phase_source
        self.normalization = str(normalization)
        self.utterance_std_floor = float(utterance_std_floor)
        if self.utterance_std_floor <= 0:
            raise ValueError("utterance_std_floor must be positive")
        if self.normalization not in ("none", "utterance"):
            raise ValueError("Spectral z-score supports none or utterance")
        if self.normalization == "utterance" and type != "log_magnitude":
            raise ValueError("Spectral utterance z-score currently requires log_magnitude")
        self.channels = 2 if type == "complex_ri" else 1

    def encode(self, waveform):
        spectrum = self.stft(waveform)
        if self.type == "complex_ri":
            return complex_to_channels(self.compress(spectrum))
        if self.type == "magnitude":
            return self.compress(spectrum).abs()[:, None]
        if self.type == "log_magnitude":
            value = torch.log(spectrum.abs() + self.eps)
            if self.normalization == "utterance":
                mu, std = self._logmag_noisy_stats(value)
                value = (value - mu) / std
            return value[:, None]
        raise RuntimeError("Unhandled representation type: " + self.type)

    def _logmag_noisy_stats(self, noisy_log):
        """Noisy-only [B,1,1] stats, separate for each batch item."""
        mean = noisy_log.mean(dim=(-2, -1), keepdim=True)
        std = noisy_log.std(dim=(-2, -1), unbiased=False,
                            keepdim=True).clamp_min(self.utterance_std_floor)
        return mean, std

    def encode_pair(self, clean_waveform, noisy_waveform):
        """Shared noisy-derived stats; essential for supervised training."""
        if clean_waveform.shape != noisy_waveform.shape:
            raise ValueError("Clean/noisy waveform shapes differ")
        if self.type != "log_magnitude" or self.normalization != "utterance":
            return self.encode(clean_waveform), self.encode(noisy_waveform)
        clean_log = torch.log(self.stft(clean_waveform).abs() + self.eps)
        noisy_log = torch.log(self.stft(noisy_waveform).abs() + self.eps)
        mu, std = self._logmag_noisy_stats(noisy_log)
        return ((clean_log - mu) / std)[:, None], ((noisy_log - mu) / std)[:, None]

    def to_spectrum(self, representation, noisy_phase=None, noisy_waveform=None):
        if representation.ndim != 4 or representation.shape[1] != self.channels:
            raise ValueError("Expected [B,C,F,T] representation")

        if self.type == "complex_ri":
            compressed = channels_to_complex(representation.float())
            return self.decompress(compressed)

        if noisy_phase is None and noisy_waveform is not None:
            noisy_phase = self.stft(noisy_waveform.float()).angle()
        if noisy_phase is None or noisy_phase.shape != representation[:, 0].shape:
            raise ValueError("Magnitude reconstruction requires aligned noisy phase")

        if self.type == "magnitude":
            # Generative states are unconstrained; clamp only at final reconstruction.
            compressed = torch.polar(representation[:, 0].float().clamp_min(0), noisy_phase)
            return self.decompress(compressed)

        if self.type == "log_magnitude":
            log_mag = representation[:, 0].float()
            if self.normalization == "utterance":
                if noisy_waveform is None:
                    raise ValueError("Utterance log_magnitude decoding requires noisy_waveform")
                noisy_log = torch.log(self.stft(noisy_waveform.float()).abs() + self.eps)
                if noisy_log.shape != log_mag.shape:
                    raise ValueError("Noisy/log_magnitude frame mismatch")
                mu, std = self._logmag_noisy_stats(noisy_log)
                log_mag = log_mag * std + mu
            # Preserve original log-magnitude exp clamp at physical reconstruction.
            magnitude = (torch.exp(log_mag.clamp(-30.0, 30.0)) - self.eps).clamp_min(0)
            return torch.polar(magnitude, noisy_phase)

        raise RuntimeError("Unhandled representation type: " + self.type)

    def decode(self, representation, length, noisy_phase=None, noisy_waveform=None):
        return self.istft(self.to_spectrum(
            representation, noisy_phase, noisy_waveform=noisy_waveform), length)
    
# """Real model channels with reversible complex-STFT boundary adapters."""
# import torch
# from .stft import ComplexSTFTTransform, complex_to_channels, channels_to_complex


# class SpectralRepresentation(ComplexSTFTTransform):
#     def __init__(self, type="complex_ri", phase_source="noisy", **kwargs):
#         super().__init__(**kwargs)
#         if type not in ("complex_ri", "magnitude"):
#             raise ValueError("Unknown spectral representation")
#         if phase_source != "noisy":
#             raise ValueError("Only noisy phase reconstruction is implemented")
#         self.type, self.phase_source = type, phase_source
#         self.channels = 2 if type == "complex_ri" else 1

#     # def encode(self, waveform):
#     #     spectrum = self.compress(self.stft(waveform))
#     #     return complex_to_channels(spectrum) if self.channels == 2 else spectrum.abs()[:, None]

#     def encode(self, waveform):
#         spectrum = self.stft(waveform)

#         if self.type == "complex_ri":
#             spectrum = self.compress(spectrum)
#             return complex_to_channels(spectrum)

#         elif self.type == "magnitude":
#             spectrum = self.compress(spectrum)
#             return spectrum.abs()[:, None]

#         elif self.type == "log_magnitude":
#             magnitude = spectrum.abs()
#             log_mag = torch.log(
#                 magnitude + self.eps
#             )
#             return log_mag[:, None]
    
#     def to_spectrum(self, representation, noisy_phase=None):
#         if representation.ndim != 4 or representation.shape[1] != self.channels:
#             raise ValueError("Expected [B,C,F,T] representation")
#         if self.channels == 2:
#             compressed = channels_to_complex(representation.float())
#         else:
#             if noisy_phase is None or noisy_phase.shape != representation[:, 0].shape:
#                 raise ValueError("Magnitude reconstruction requires aligned noisy phase")
#             # This is the ONLY magnitude clamp: generative states remain unconstrained.
#             compressed = torch.polar(representation[:, 0].float().clamp_min(0), noisy_phase)

#         return self.decompress(compressed)

#     def decode(self, representation, length, noisy_phase=None):
#         return self.istft(self.to_spectrum(representation, noisy_phase), length)

