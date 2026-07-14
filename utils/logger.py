"""Console/file logging and optional TensorBoard integration."""

import logging
from pathlib import Path
from typing import Optional


def create_logger(name: str, output_dir: str) -> logging.Logger:
    """Create an idempotent logger writing to console and ``train.log``."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(directory / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def create_summary_writer(output_dir: str) -> Optional[object]:
    """Create a TensorBoard writer when tensorboard is installed."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except (ImportError, ModuleNotFoundError):
        return None
    return SummaryWriter(log_dir=str(Path(output_dir) / "tensorboard"))

