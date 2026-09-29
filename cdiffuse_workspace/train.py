"""Train CDiffuSE with SGMSE-aligned logging and generative validation."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from cdiffuse.data import CDiffuSEDataset, build_dataloader
from cdiffuse.engine import Trainer
from cdiffuse.factory import build_components
from cdiffuse.utils.config import load_config
from cdiffuse.utils.ema import EMA
from cdiffuse.utils.logging import JsonlLogger, create_logger, create_summary_writer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["training"]["seed"])
    seed_everything(seed)

    device = torch.device(args.device)
    output_dir = str(Path(config["project"]["output_dir"]).resolve())
    config["project"]["output_dir"] = output_dir

    logger = create_logger(
        config["project"]["name"],
        output_dir,
        enabled=True,
        filename="training.log",
    )

    train_dataset = CDiffuSEDataset(config["data"], "train", seed)
    valid_dataset = CDiffuSEDataset(config["data"], "valid", seed + 1)

    train_loader = build_dataloader(train_dataset, config, True)
    valid_loader = build_dataloader(valid_dataset, config, False)
    # Separate deterministic-order loader for expensive full sampling validation.
    sampling_valid_loader = build_dataloader(valid_dataset, config, False)

    model, schedule, sampler = build_components(config)
    model.to(device)

    ema = EMA(model, float(config["training"]["ema_decay"]))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["training"].get("scheduler_factor", 0.5)),
        patience=int(config["training"].get("scheduler_patience", 5)),
        min_lr=float(config["training"].get("min_lr", 1.0e-7)),
    )

    logger.info(
        "device=%s train=%d valid=%d parameters=%d diffusion_steps=%d",
        device,
        len(train_dataset),
        len(valid_dataset),
        sum(parameter.numel() for parameter in model.parameters()),
        schedule.num_steps,
    )

    logging_cfg = config.get("logging", {})
    trainer = Trainer(
        model,
        schedule,
        sampler,
        optimizer,
        scheduler,
        ema,
        config,
        device,
        logger,
        JsonlLogger(
            str(Path(output_dir) / "metrics.jsonl"),
            bool(logging_cfg.get("jsonl", True)),
        ),
        create_summary_writer(
            output_dir,
            bool(logging_cfg.get("tensorboard", True)),
        ),
    )

    resume = args.resume or config["training"].get("resume")
    if resume:
        trainer.resume(resume)

    trainer.fit(train_loader, valid_loader, sampling_valid_loader)


if __name__ == "__main__":
    main()
