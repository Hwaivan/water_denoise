"""Generate enhanced test WAVs and explicit SDR/SI-SNR reports."""

import argparse
import json
from pathlib import Path

import torch

from dit.data import AudioDataset, build_dataloader
from dit.engine.evaluator import evaluate
from dit.factory import build_components
from dit.utils.checkpoint import load_model_for_inference
from dit.utils.config import load_config, add_evaluation_arguments, apply_evaluation_arguments
from dit.utils.distributed import (
    barrier,
    cleanup_distributed,
    initialize_distributed,
)
from dit.utils.logging import create_logger
from dit.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Conditional DiT checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="")
    add_evaluation_arguments(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    apply_evaluation_arguments(config, args)
    context = initialize_distributed(args.device)
    try:
        if not context.is_main:
            barrier(context)
            return
        seed = args.seed if args.seed is not None else int(config["validation"].get("seed", 1234))
        seed_everything(seed, True)
        model, _, _, _, sampler = build_components(config)
        model = model.to(context.device).eval()
        load_model_for_inference(
            args.checkpoint,
            model,
            str(context.device),
            bool(config["sampler"].get("use_ema", True)),
            config=config,
        )
        dataset = AudioDataset(config["data"], args.split, seed)
        loader = build_dataloader(dataset, config, False, seed, False)
        logger = create_logger(
            "dit_evaluation",
            args.output_dir,
            True,
            filename="evaluation.log",
        )
        try:
            generator = torch.Generator(device=context.device)
        except TypeError:
            generator = torch.Generator(device=context.device.type)
        generator.manual_seed(seed)
        summary = evaluate(
            model,
            sampler,
            loader,
            context.device,
            config,
            args.checkpoint,
            args.output_dir,
            seed,
            generator,
            logger,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        barrier(context)
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()


