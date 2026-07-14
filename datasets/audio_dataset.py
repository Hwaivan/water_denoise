"""Separate-list online mixing and fixed-pair speech enhancement datasets."""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .audio_io import load_audio
from .mixing import crop_or_pad, mix_at_snr, remove_dc


def _read_lines(path: str) -> List[str]:
    list_path = Path(path).expanduser()
    if not list_path.is_file():
        raise FileNotFoundError("List file not found: {}".format(list_path))
    with list_path.open("r", encoding="utf-8-sig") as stream:
        lines = [line.strip() for line in stream if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise ValueError("List file is empty: {}".format(list_path))
    return lines


class SpeechEnhancementDataset(Dataset):
    """Single-channel supervised enhancement dataset.

    ``separate_lists`` pairs each clean item with a deterministic-per-epoch random
    noise item and SNR. ``paired`` expects tab-separated ``noisy<TAB>clean`` rows.
    """

    def __init__(
        self,
        mode: str,
        sample_rate: int,
        segment_seconds: Optional[float] = 4.0,
        clean_list: Optional[str] = None,
        noise_list: Optional[str] = None,
        pair_list: Optional[str] = None,
        snr_min: float = -5.0,
        snr_max: float = 20.0,
        training: bool = True,
        seed: int = 42,
        random_gain: bool = False,
        polarity_flip: bool = False,
        time_shift: bool = False,
    ) -> None:
        self.mode = mode
        self.sample_rate = int(sample_rate)
        self.segment_length = (
            int(round(segment_seconds * sample_rate)) if segment_seconds is not None else None
        )
        self.snr_min = float(snr_min)
        self.snr_max = float(snr_max)
        self.training = bool(training)
        self.seed = int(seed)
        self.epoch = 0
        self.random_gain = bool(random_gain)
        self.polarity_flip = bool(polarity_flip)
        self.time_shift = bool(time_shift)
        if mode == "separate_lists":
            if clean_list is None or noise_list is None:
                raise ValueError("separate_lists mode requires clean_list and noise_list")
            self.clean_paths = _read_lines(clean_list)
            self.noise_paths = _read_lines(noise_list)
            self.pairs = []
        elif mode == "paired":
            if pair_list is None:
                raise ValueError("paired mode requires pair_list")
            self.pairs = []
            for row in _read_lines(pair_list):
                fields = row.split("\t")
                if len(fields) != 2:
                    raise ValueError("Pair rows must be noisy<TAB>clean: {}".format(row))
                self.pairs.append((fields[0], fields[1]))
            self.clean_paths, self.noise_paths = [], []
        else:
            raise ValueError("Unknown dataset mode: {}".format(mode))

    def set_epoch(self, epoch: int) -> None:
        """Set epoch so online mixtures vary reproducibly across epochs."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.clean_paths) if self.mode == "separate_lists" else len(self.pairs)

    def _generator(self, index: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * max(1, len(self)) + index)
        return generator

    def _fit(self, waveform: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        if self.segment_length is None:
            return waveform
        return crop_or_pad(waveform, self.segment_length, generator, self.training)

    def __getitem__(self, index: int) -> Dict[str, object]:
        generator = self._generator(index)
        if self.mode == "separate_lists":
            target_path = self.clean_paths[index]
            noise_index = int(torch.randint(len(self.noise_paths), (1,), generator=generator).item())
            noise_path = self.noise_paths[noise_index]
            target, _ = load_audio(target_path, self.sample_rate)
            noise, _ = load_audio(noise_path, self.sample_rate)
            target = self._fit(target, generator)
            noise = self._fit(noise, generator)
            if self.time_shift and noise.numel() > 1:
                shift = int(torch.randint(noise.numel(), (1,), generator=generator).item())
                noise = torch.roll(noise, shift)
            if self.polarity_flip and torch.rand((), generator=generator) < 0.5:
                target = -target
            if self.random_gain:
                gain = float(torch.empty(1).uniform_(0.5, 1.0, generator=generator).item())
                target = target * gain
            snr = float(
                torch.empty(1).uniform_(self.snr_min, self.snr_max, generator=generator).item()
            )
            mixture, noise = mix_at_snr(target, noise, snr)
            mixture_path = noise_path
        else:
            mixture_path, target_path = self.pairs[index]
            mixture, _ = load_audio(mixture_path, self.sample_rate)
            target, _ = load_audio(target_path, self.sample_rate)
            common = min(mixture.numel(), target.numel())
            mixture, target = mixture[:common], target[:common]
            if self.segment_length is not None:
                max_start = max(0, common - self.segment_length)
                start = (
                    int(torch.randint(max_start + 1, (1,), generator=generator).item())
                    if self.training and max_start
                    else 0
                )
                mixture = crop_or_pad(mixture[start:], self.segment_length, generator, False)
                target = crop_or_pad(target[start:], self.segment_length, generator, False)
            mixture, target = remove_dc(mixture), remove_dc(target)
            noise = mixture - target
            signal_power = target.square().mean().clamp_min(1.0e-8)
            noise_power = noise.square().mean().clamp_min(1.0e-8)
            snr = float((10.0 * torch.log10(signal_power / noise_power)).item())
        length = int(mixture.numel())
        return {
            "mixture": mixture,
            "target": target,
            "noise": noise,
            "length": length,
            "mixture_path": str(mixture_path),
            "target_path": str(target_path),
            "input_snr": snr,
        }


def collate_audio_batch(items: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Pad variable-length dataset items into a batch."""
    if not items:
        raise ValueError("Cannot collate an empty batch")
    result = {
        key: pad_sequence([item[key] for item in items], batch_first=True)
        for key in ("mixture", "target", "noise")
    }
    result["lengths"] = torch.tensor([item["length"] for item in items], dtype=torch.long)
    result["mixture_path"] = [item["mixture_path"] for item in items]
    result["target_path"] = [item["target_path"] for item in items]
    result["input_snr"] = torch.tensor([item["input_snr"] for item in items], dtype=torch.float32)
    return result

