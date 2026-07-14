"""Evaluate a DCCRN checkpoint on the configured test pairs."""

import argparse
import json

import torch

from datasets.factory import build_dataset, build_loader
from engine.evaluator import evaluate_dataset
from models import build_model
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DCCRN")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config["experiment"].get("seed", 42))
    seed_everything(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, device=str(device))
    dataset = build_dataset(config, "test", seed + 2)
    loader = build_loader(dataset, config, False, seed + 2)
    output_dir = args.output_dir or str(config["experiment"]["output_dir"] + "/evaluation")
    evaluation = config.get("evaluation", {})
    summary = evaluate_dataset(
        model,
        loader,
        device,
        output_dir,
        sample_rate=int(config["data"]["sample_rate"]),
        save_audio_examples=bool(evaluation.get("save_audio_examples", False)),
        max_audio_examples=int(evaluation.get("max_audio_examples", 10)),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
