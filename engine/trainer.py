"""Native-PyTorch training loop with AMP, checkpoints, and early stopping."""

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from torch import nn

from losses.si_snr import LossManager
from metrics.enhancement_metrics import compute_metrics
from utils.checkpoint import load_checkpoint, save_checkpoint


class Trainer:
    """Own the complete train/validation state machine."""

    def __init__(
        self,
        model: nn.Module,
        loss_manager: LossManager,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        device: torch.device,
        config: Dict[str, Any],
        logger: logging.Logger,
        writer: Optional[Any] = None,
    ) -> None:
        self.model = model.to(device)
        self.loss_manager = loss_manager.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.logger = logger
        self.writer = writer
        training = config["training"]
        self.amp_enabled = bool(training.get("amp", True) and device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.gradient_clip = float(training.get("gradient_clip_norm", 0.0))
        self.output_dir = Path(config["experiment"]["output_dir"])
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.start_epoch = 0
        self.best_metric = -math.inf
        self.bad_epochs = 0
        self._validation_example = None

    def resume(self, path: str) -> None:
        """Restore all available training state."""
        checkpoint = load_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            str(self.device),
        )
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        self.best_metric = float(checkpoint.get("best_metric", -math.inf))
        self.bad_epochs = int(checkpoint.get("bad_epochs", 0))
        self.logger.info("Resumed %s at epoch %d", path, self.start_epoch)

    def _batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {
            key: batch[key].to(self.device, non_blocking=True)
            for key in ("mixture", "target", "lengths")
        }

    def train_one_epoch(self, loader: Iterable[Dict[str, Any]], epoch: int) -> Dict[str, float]:
        """Run one optimization epoch and return sample-weighted means."""
        self.model.train()
        totals = {"loss": 0.0, "si_snr": 0.0, "grad_norm": 0.0, "samples": 0.0}
        for batch_index, raw_batch in enumerate(loader):
            batch = self._batch(raw_batch)
            batch_size = batch["mixture"].shape[0]
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                outputs = self.model(batch["mixture"], batch["lengths"])
                losses = self.loss_manager(outputs, batch["target"], batch["lengths"])
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(
                    "Non-finite loss at epoch {}, batch {}".format(epoch, batch_index)
                )
            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            if self.gradient_clip > 0:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
            else:
                norms = [
                    parameter.grad.detach().norm(2)
                    for parameter in self.model.parameters()
                    if parameter.grad is not None
                ]
                gradient_norm = torch.stack(norms).norm(2) if norms else losses["total"].new_zeros(())
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite gradient norm")
            self.scaler.step(self.optimizer)
            self.scaler.update()
            totals["loss"] += float(losses["total"].detach()) * batch_size
            totals["si_snr"] += float(losses["si_snr"].detach()) * batch_size
            totals["grad_norm"] += float(gradient_norm) * batch_size
            totals["samples"] += batch_size
        count = max(1.0, totals.pop("samples"))
        return {key: value / count for key, value in totals.items()}

    @torch.inference_mode()
    def validate(self, loader: Iterable[Dict[str, Any]]) -> Dict[str, float]:
        """Run deterministic validation using the same model forward interface."""
        self.model.eval()
        totals = {"loss": 0.0, "si_snr": 0.0, "samples": 0.0}
        for batch_index, raw_batch in enumerate(loader):
            batch = self._batch(raw_batch)
            outputs = self.model(batch["mixture"], batch["lengths"])
            losses = self.loss_manager(outputs, batch["target"], batch["lengths"])
            metrics = compute_metrics(
                outputs["waveform"], batch["target"], batch["mixture"], batch["lengths"]
            )
            if batch_index == 0:
                valid_length = int(batch["lengths"][0].item())
                self._validation_example = {
                    "mixture": batch["mixture"][0, :valid_length].detach().cpu(),
                    "target": batch["target"][0, :valid_length].detach().cpu(),
                    "enhanced": outputs["waveform"][0, :valid_length].detach().cpu(),
                }
            batch_size = batch["mixture"].shape[0]
            totals["loss"] += float(losses["total"]) * batch_size
            totals["si_snr"] += float(metrics["si_snr"].mean()) * batch_size
            totals["samples"] += batch_size
        count = max(1.0, totals.pop("samples"))
        return {key: value / count for key, value in totals.items()}

    def _state(self, epoch: int) -> Dict[str, Any]:
        return {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "bad_epochs": self.bad_epochs,
            "config": self.config,
        }

    def fit(self, train_loader: Any, valid_loader: Any) -> None:
        """Train until configured epochs or early stopping."""
        training = self.config["training"]
        epochs = int(training["epochs"])
        save_every = int(training.get("save_every", 5))
        patience = int(training.get("early_stopping_patience", 10))
        for epoch in range(self.start_epoch, epochs):
            started = time.time()
            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)
            train_stats = self.train_one_epoch(train_loader, epoch)
            valid_stats = self.validate(valid_loader)
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(valid_stats["si_snr"])
                else:
                    self.scheduler.step()
            improved = valid_stats["si_snr"] > self.best_metric
            if improved:
                self.best_metric = valid_stats["si_snr"]
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
            elapsed = time.time() - started
            learning_rate = self.optimizer.param_groups[0]["lr"]
            gpu_memory = (
                torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 2)
                if self.device.type == "cuda"
                else 0.0
            )
            self.logger.info(
                "epoch=%d train_loss=%.4f train_si_snr=%.3f val_loss=%.4f "
                "val_si_snr=%.3f lr=%.3e grad_norm=%.3f time=%.1fs gpu_mb=%.1f",
                epoch + 1,
                train_stats["loss"],
                train_stats["si_snr"],
                valid_stats["loss"],
                valid_stats["si_snr"],
                learning_rate,
                train_stats["grad_norm"],
                elapsed,
                gpu_memory,
            )
            if self.writer is not None:
                for name, value in train_stats.items():
                    self.writer.add_scalar("train/{}".format(name), value, epoch + 1)
                for name, value in valid_stats.items():
                    self.writer.add_scalar("valid/{}".format(name), value, epoch + 1)
                self.writer.add_scalar("train/learning_rate", learning_rate, epoch + 1)
                self._write_validation_example(epoch + 1)
            state = self._state(epoch)
            save_checkpoint(state, str(self.checkpoint_dir / "last.pt"))
            if improved:
                save_checkpoint(state, str(self.checkpoint_dir / "best.pt"))
            if save_every > 0 and (epoch + 1) % save_every == 0:
                save_checkpoint(state, str(self.checkpoint_dir / "epoch_{:04d}.pt".format(epoch + 1)))
            if self.bad_epochs >= patience:
                self.logger.info("Early stopping after %d epochs without improvement", patience)
                break
        if self.writer is not None:
            self.writer.close()

    def _write_validation_example(self, step: int) -> None:
        """Write one validation audio triplet and its log-magnitude spectra."""
        if self.writer is None or self._validation_example is None:
            return
        sample_rate = int(self.config["data"]["sample_rate"])
        for name, waveform in self._validation_example.items():
            peak = waveform.abs().max().clamp_min(1.0)
            self.writer.add_audio("audio/{}".format(name), waveform / peak, step, sample_rate)
            spectrum = self.model.frontend.transform(waveform.view(1, -1).to(self.device))
            image = torch.log1p(spectrum.abs())[0].detach().cpu()
            image = image / image.max().clamp_min(1.0e-8)
            self.writer.add_image("spectrogram/{}".format(name), image.unsqueeze(0), step)
