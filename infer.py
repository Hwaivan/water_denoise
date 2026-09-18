"""Single-file and overlap-add long-audio DCCRN inference."""

import argparse
from typing import Optional

import torch

from datasets.audio_io import load_audio, save_audio
from models import build_model
from utils.checkpoint import load_checkpoint
from utils.config import load_config


@torch.inference_mode()
def enhance_waveform(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    device: torch.device,
    chunk_samples: Optional[int] = None,
    overlap_samples: int = 0,
) -> torch.Tensor:
    """Enhance a waveform, using weighted overlap-add when chunking is enabled."""
    original_length = waveform.numel()
    if chunk_samples is None or original_length <= chunk_samples:
        lengths = torch.tensor([original_length], device=device)
        return model(waveform.view(1, -1).to(device), lengths)["waveform"][0].cpu()
    if overlap_samples < 0 or overlap_samples >= chunk_samples:
        raise ValueError("overlap_samples must be in [0, chunk_samples)")
    step = chunk_samples - overlap_samples
    output = torch.zeros(original_length, device=device)
    weights = torch.zeros(original_length, device=device)
    starts = list(range(0, original_length, step))
    for chunk_index, start in enumerate(starts):
        valid = min(chunk_samples, original_length - start)
        chunk = waveform[start : start + valid]
        if valid < chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - valid))
        enhanced = model(
            chunk.view(1, -1).to(device), torch.tensor([valid], device=device)
        )["waveform"][0, :valid]
        window = torch.ones(valid, device=device)
        fade = min(overlap_samples, valid)
        if fade and chunk_index > 0:
            window[:fade] = torch.linspace(0.0, 1.0, fade, device=device)
        if fade and start + valid < original_length:
            window[-fade:] = torch.minimum(
                window[-fade:], torch.linspace(1.0, 0.0, fade, device=device)
            )
        output[start : start + valid] += enhanced * window
        weights[start : start + valid] += window
        if start + valid >= original_length:
            break
    return (output / weights.clamp_min(1.0e-8)).cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhance one audio file with DCCRN")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--chunk-seconds", type=float, default=None)
    parser.add_argument("--overlap-seconds", type=float, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    sample_rate = int(config["data"]["sample_rate"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config).to(device).eval()
    load_checkpoint(args.checkpoint, model, device=str(device))
    waveform, _ = load_audio(args.input, sample_rate)
    inference = config.get("inference", {})
    chunk_seconds = args.chunk_seconds or inference.get("chunk_seconds", 10.0)
    overlap_seconds = args.overlap_seconds
    if overlap_seconds is None:
        overlap_seconds = inference.get("overlap_seconds", 1.0)
    enhanced = enhance_waveform(
        model,
        waveform,
        device,
        int(round(chunk_seconds * sample_rate)) if chunk_seconds else None,
        int(round(overlap_seconds * sample_rate)),
    )
    save_audio(args.output, enhanced[: waveform.numel()], sample_rate)


if __name__ == "__main__":
    main()

