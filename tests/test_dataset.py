"""Tests for deterministic online mixing and batching."""

from pathlib import Path

import torch

from datasets.audio_dataset import SpeechEnhancementDataset, collate_audio_batch


def test_online_mixture_has_requested_snr(tmp_path: Path, monkeypatch) -> None:
    clean_list = tmp_path / "clean.txt"
    noise_list = tmp_path / "noise.txt"
    clean_list.write_text("clean.wav\n", encoding="utf-8")
    noise_list.write_text("noise.wav\n", encoding="utf-8")
    clean = torch.sin(torch.linspace(0, 100, 16000))
    noise = torch.cos(torch.linspace(0, 371, 16000))

    def fake_load(path: str, sample_rate: int):
        return (clean.clone() if path == "clean.wav" else noise.clone()), sample_rate

    monkeypatch.setattr("datasets.audio_dataset.load_audio", fake_load)
    dataset = SpeechEnhancementDataset(
        mode="separate_lists",
        clean_list=str(clean_list),
        noise_list=str(noise_list),
        sample_rate=16000,
        segment_seconds=1.0,
        snr_min=5.0,
        snr_max=5.0,
        training=False,
    )
    item = dataset[0]
    actual_snr = 10.0 * torch.log10(
        item["target"].square().mean() / item["noise"].square().mean()
    )
    assert abs(float(actual_snr) - 5.0) < 0.05
    batch = collate_audio_batch([item, item])
    assert batch["mixture"].shape == (2, 16000)
    assert batch["lengths"].tolist() == [16000, 16000]


def test_online_mixture_is_reproducible_per_epoch(tmp_path: Path, monkeypatch) -> None:
    clean_list = tmp_path / "clean.txt"
    noise_list = tmp_path / "noise.txt"
    clean_list.write_text("clean.wav\n", encoding="utf-8")
    noise_list.write_text("noise.wav\n", encoding="utf-8")
    monkeypatch.setattr(
        "datasets.audio_dataset.load_audio",
        lambda path, rate: (torch.arange(2000, dtype=torch.float32), rate),
    )
    dataset = SpeechEnhancementDataset(
        "separate_lists", 1000, 1.0, str(clean_list), str(noise_list), seed=7
    )
    first = dataset[0]["mixture"]
    second = dataset[0]["mixture"]
    assert torch.equal(first, second)

