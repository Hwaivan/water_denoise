"""Command-line entry point for DCCRN training."""

import argparse
from pathlib import Path

import torch

from datasets.factory import build_dataset, build_loader
from engine.trainer import Trainer
from losses import LossManager
from models import build_model
from utils.config import load_config, require_sections
from utils.logger import create_logger, create_summary_writer
from utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train single-channel DCCRN")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None, choices=("cpu", "cuda"))
    return parser.parse_args()


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    training = config["training"]
    if training.get("optimizer", "adam").lower() != "adam":
        raise ValueError("Only Adam is currently supported")
    return torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    scheduler = config.get("scheduler", {})
    name = scheduler.get("name", "reduce_on_plateau").lower()
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(scheduler.get("factor", 0.5)),
            patience=int(scheduler.get("patience", 2)),
            min_lr=float(scheduler.get("min_lr", 1.0e-6)),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(config["training"]["epochs"]), eta_min=float(scheduler.get("min_lr", 1.0e-6))
        )
    if name in ("none", "null"):
        return None
    raise ValueError("Unknown scheduler: {}".format(name))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    require_sections(config, "experiment", "data", "stft", "model", "loss", "training")
    seed = int(config["experiment"].get("seed", 42))
    seed_everything(seed, bool(config["experiment"].get("deterministic", True)))
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    output_dir = config["experiment"]["output_dir"]
    logger = create_logger(config["experiment"]["name"], output_dir)
    model = build_model(config)
    train_dataset = build_dataset(config, "train", seed)
    valid_dataset = build_dataset(config, "valid", seed + 1)
    train_loader = build_loader(train_dataset, config, True, seed)
    valid_loader = build_loader(valid_dataset, config, False, seed + 1)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    logger.info(
        "device=%s parameters=%d trainable=%d train_items=%d valid_items=%d",
        device,
        total_parameters,
        trainable_parameters,
        len(train_dataset),
        len(valid_dataset),
    )
    optimizer = build_optimizer(model, config)
    trainer = Trainer(
        model,
        LossManager(config["loss"]),
        optimizer,
        build_scheduler(optimizer, config),
        device,
        config,
        logger,
        create_summary_writer(output_dir),
    )
    resume = args.resume or config["training"].get("resume")
    if resume:
        trainer.resume(resume)
    trainer.fit(train_loader, valid_loader)


if __name__ == "__main__":
    main()
