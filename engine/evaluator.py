"""Dataset evaluation with per-sample CSV and aggregate JSON output."""

import csv
import json
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List

import torch

from datasets.audio_io import save_audio
from metrics.enhancement_metrics import compute_metrics


@torch.inference_mode()
def evaluate_dataset(
    model: torch.nn.Module,
    loader: Iterable[Dict[str, Any]],
    device: torch.device,
    output_dir: str,
    sample_rate: int = 16000,
    save_audio_examples: bool = False,
    max_audio_examples: int = 10,
) -> Dict[str, Any]:
    """Evaluate a loader and persist per-sample and aggregate results."""
    model.eval()
    rows: List[Dict[str, Any]] = []
    for batch in loader:
        mixture = batch["mixture"].to(device)
        target = batch["target"].to(device)
        lengths = batch["lengths"].to(device)
        outputs = model(mixture, lengths)
        metrics = compute_metrics(outputs["waveform"], target, mixture, lengths)
        for index in range(mixture.shape[0]):
            row = {
                "mixture_path": batch["mixture_path"][index],
                "target_path": batch["target_path"][index],
                "input_snr": float(batch["input_snr"][index]),
            }
            row.update({name: float(values[index].cpu()) for name, values in metrics.items()})
            if save_audio_examples and len(rows) < max_audio_examples:
                valid_length = int(lengths[index].item())
                audio_dir = Path(output_dir) / "audio_examples"
                stem = "{:04d}".format(len(rows))
                save_audio(
                    str(audio_dir / (stem + "_mixture.wav")),
                    mixture[index, :valid_length],
                    sample_rate,
                )
                save_audio(
                    str(audio_dir / (stem + "_target.wav")),
                    target[index, :valid_length],
                    sample_rate,
                )
                save_audio(
                    str(audio_dir / (stem + "_enhanced.wav")),
                    outputs["waveform"][index, :valid_length],
                    sample_rate,
                )
            rows.append(row)
    if not rows:
        raise ValueError("Evaluation dataset is empty")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    fields = ["mixture_path", "target_path", "input_snr", "si_snr", "si_snri", "sdr", "sdri", "l1", "corr"]
    with (directory / "per_sample.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metric_names = fields[3:]
    summary: Dict[str, Any] = {
        name: {
            "mean": mean([row[name] for row in rows]),
            "median": median([row[name] for row in rows]),
            "std": pstdev([row[name] for row in rows]),
        }
        for name in metric_names
    }
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        bucket = "{}dB".format(int(round(row["input_snr"] / 5.0) * 5))
        grouped.setdefault(bucket, []).append(row)
    summary["by_input_snr"] = {
        bucket: {name: mean([row[name] for row in group]) for name in metric_names}
        for bucket, group in sorted(grouped.items())
    }
    with (directory / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary
