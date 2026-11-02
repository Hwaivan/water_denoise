#!/usr/bin/env python3
"""Estimate fixed per-Mel-band mean/std from the TRAIN split, without leakage.

Uses the existing AudioDataset so on_the_fly mixing, alignment, limiting and
paired data match the actual training data pipeline. Outputs a hash into YAML.
"""
import argparse
import hashlib
from pathlib import Path

import torch
import yaml

from dit.representations.mel import LogMelRepresentation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--max-items", type=int, default=0, help="0=entire train dataset")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    path = Path(args.config).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if cfg["representation"]["type"] == "log_mel":
        mel = cfg["representation"]["mel"]
    else:
        mel = cfg["conditioning"]["mel"]
    if mel.get("normalization", "dataset") != "dataset":
        raise ValueError("Only dataset normalization needs offline Mel stats")
    stats = Path(mel["stats_path"]).resolve()
    mel_temp = dict(mel)
    mel_temp["normalization"] = "none"
    mel_temp.pop("stats_path", None)
    mel_temp.pop("stats_sha256", None)
    compression = {k: v for k, v in cfg["compression"].items() if k != "crop_frames"}
    codec = LogMelRepresentation(mel=mel_temp,
                                 sample_rate=int(cfg["data"]["sample_rate"]),
                                 **cfg["stft"], **compression).eval()
    # Import the actual dataset AFTER model-independent setup for a clear error message.
    try:
        from dit.data.dataset import AudioDataset
    except ImportError as exc:
        raise RuntimeError("Existing dit.data.dataset.AudioDataset is required to prepare stats") from exc
    dataset = AudioDataset(cfg["data"], "train", args.seed)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(0)
    n_items = len(dataset) if args.max_items <= 0 else min(len(dataset), args.max_items)
    if n_items < 1:
        raise RuntimeError("No train samples available for Mel statistics")
    sums = torch.zeros(codec.n_mels, dtype=torch.float64)
    sums2 = sums.clone()
    n_frames = 0
    with torch.inference_mode():
        for i in range(n_items):
            item = dataset[i]
            n = int(item.get("length", item["clean"].numel()))
            for key in ("clean", "noisy"):
                wave = item[key].float().reshape(1, -1)[:, :n]
                values = codec.log_mel(wave)[0].double()
                sums += values.sum(dim=-1)
                sums2 += values.square().sum(dim=-1)
                n_frames += values.shape[-1]
            if (i + 1) % 250 == 0:
                print(f"Mel stats {i+1}/{n_items}", flush=True)
    mean = sums / n_frames
    variance = (sums2 / n_frames - mean.square()).clamp_min(0)
    std = variance.sqrt().clamp_min(1e-3)
    result = {"mean": mean.float(), "std": std.float(),
              "frame_count": torch.tensor(n_frames),
              "n_items": torch.tensor(n_items),
              "n_mels": torch.tensor(codec.n_mels)}
    stats.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, stats)
    digest = hashlib.sha256(stats.read_bytes()).hexdigest()
    mel["stats_path"] = str(stats)
    mel["stats_sha256"] = digest
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Saved {stats}: {n_items} items, {n_frames} frames, sha256={digest}")
    print(f"Range mean={float(mean.min()):.3f}..{float(mean.max()):.3f}, "
          f"std={float(std.min()):.3f}..{float(std.max()):.3f}")


if __name__ == "__main__":
    main()
