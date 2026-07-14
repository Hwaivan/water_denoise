"""Dataset and DataLoader construction from experiment configuration."""

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from utils.seed import seed_worker
from .audio_dataset import SpeechEnhancementDataset, collate_audio_batch


def build_dataset(config: Dict[str, Any], split: str, seed: int) -> SpeechEnhancementDataset:
    """Build a train/valid/test dataset from the ``data`` section."""
    data = config["data"]
    training = split == "train"
    if training and data["mode"] == "separate_lists":
        return SpeechEnhancementDataset(
            mode="separate_lists",
            clean_list=data["train_clean_list"],
            noise_list=data["train_noise_list"],
            sample_rate=data["sample_rate"],
            segment_seconds=data.get("segment_seconds", 4.0),
            snr_min=data.get("train_snr_min", -5.0),
            snr_max=data.get("train_snr_max", 20.0),
            training=True,
            seed=seed,
            random_gain=data.get("augment", {}).get("random_gain", False),
            polarity_flip=data.get("augment", {}).get("polarity_flip", False),
            time_shift=data.get("augment", {}).get("time_shift", False),
        )
    pair_key = "{}_pair_list".format(split)
    if not data.get(pair_key):
        raise ValueError("data.{} is required for the {} split".format(pair_key, split))
    return SpeechEnhancementDataset(
        mode="paired",
        pair_list=data[pair_key],
        sample_rate=data["sample_rate"],
        segment_seconds=data.get("segment_seconds") if training else None,
        training=training,
        seed=seed,
    )


def build_loader(
    dataset: SpeechEnhancementDataset,
    config: Dict[str, Any],
    training: bool,
    seed: int,
) -> DataLoader:
    """Build a reproducible DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=training,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_audio_batch,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=(int(config["data"].get("num_workers", 0)) > 0),
    )

