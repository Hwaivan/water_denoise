"""AMP/DDP/EMA generative trainer with deterministic generative validation."""

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from torch import nn

from dit.objectives.regression import RegressionObjective
from dit.samplers.base import WaveformSampler, enhance_configured
from dit.metrics.audio_metrics import compute_audio_metrics
from dit.utils.checkpoint import (
    restore_training_state,
    save_checkpoint,
    training_state,
    validate_checkpoint_config,
)
from dit.utils.distributed import (
    DistributedContext,
    barrier,
    reduce_mean,
    unwrap_model,
)
from dit.utils.ema import ExponentialMovingAverage
from dit.utils.logging import JsonlLogger
from dit.utils.seed import capture_rng_state, restore_rng_state


def _device_generator(device: torch.device, seed: int) -> torch.Generator:
    """Create a generator on the same device as random tensors."""
    try:
        generator = torch.Generator(device=device)
    except TypeError:
        generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


class DiTTrainer:
    """Complete shared generative training state machine."""

    def __init__(
        self,
        model: nn.Module,
        objective: RegressionObjective,
        sampler: WaveformSampler,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        ema: ExponentialMovingAverage,
        context: DistributedContext,
        config: Dict[str, Any],
        logger: logging.Logger,
        jsonl_logger: JsonlLogger,
        writer: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.objective = objective.to(context.device)
        self.sampler = sampler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.context = context
        self.config = config
        self.logger = logger
        self.jsonl = jsonl_logger
        self.writer = writer
        training = config["training"]
        self.amp_enabled = bool(
            training.get("amp", True) and context.device.type == "cuda"
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.gradient_clip = float(training.get("gradient_clip_norm", 1.0))
        self.output_dir = Path(config["experiment"]["output_dir"])
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.start_epoch = 0
        self.global_step = 0
        self.best_metric = -math.inf
        self.resume_payload = None

    def resume(self, path: str) -> None:
        """Restore full training state and resume at the following epoch."""
        checkpoint = restore_training_state(
            path,
            unwrap_model(self.model),
            self.ema,
            self.optimizer,
            self.scheduler,
            self.scaler,
            str(self.context.device),
            restore_rng=False,
        )
        self.start_epoch = int(checkpoint["epoch"]) + 1
        validate_checkpoint_config(checkpoint["config"], self.config)
        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_metric = float(checkpoint.get("best_metric", -math.inf))
        self.resume_payload = checkpoint
        self.logger.info(
            "Resumed %s at epoch=%d global_step=%d",
            path,
            self.start_epoch,
            self.global_step,
        )

    def _move(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {
            name: batch[name].to(self.context.device, non_blocking=True)
            for name in ("clean", "noisy", "lengths")
        }

    def train_one_epoch(
        self, loader: Iterable[Dict[str, Any]], epoch: int
    ) -> Dict[str, float]:
        """Run one epoch and report loss, grad norm, LR, sigma and time means."""
        self.model.train()
        totals = {
            "loss": 0.0,
            "grad_norm": 0.0,
            "sigma_mean": 0.0,
            "t_mean": 0.0,
            "batches": 0.0,
        }
        for batch_index, raw_batch in enumerate(loader):
            batch = self._move(raw_batch)
            self.optimizer.zero_grad(set_to_none=True)
            try:
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    losses = self.objective(
                        self.model, batch["clean"], batch["noisy"]
                    )
            except RuntimeError as error:
                if "out of memory" in str(error).lower():
                    raise RuntimeError(
                        "CUDA OOM at epoch {}, batch {}; reduce batch_size, "
                        "crop_frames, hidden_size, or depth".format(
                            epoch, batch_index
                        )
                    ) from error
                raise
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(
                    "Non-finite objective loss at epoch {}, batch {}".format(
                        epoch, batch_index
                    )
                )
            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite gradient norm")
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update(unwrap_model(self.model))
            self.global_step += 1
            totals["loss"] += float(losses["total"].detach())
            totals["grad_norm"] += float(gradient_norm)
            totals["sigma_mean"] += float(losses.get("sigma_mean", 0.0))
            totals["t_mean"] += float(losses["t_mean"])
            totals["batches"] += 1.0
        count = max(1.0, totals.pop("batches"))
        local = {
            name: torch.tensor(value / count, device=self.context.device)
            for name, value in totals.items()
        }
        reduced = {
            name: float(reduce_mean(value, self.context).cpu())
            for name, value in local.items()
        }
        reduced["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return reduced

    @torch.inference_mode()
    def validate_objective(
        self, loader: Iterable[Dict[str, Any]], seed: int
    ) -> float:
        """Compute deterministic objective loss with fixed times and noise stream."""
        self.model.eval()
        generator = _device_generator(self.context.device, seed + self.context.rank)
        total, batches = 0.0, 0
        for raw_batch in loader:
            batch = self._move(raw_batch)
            losses = self.objective(
                self.model,
                batch["clean"],
                batch["noisy"],
                generator=generator,
                deterministic=True,
            )
            total += float(losses["total"])
            batches += 1
        value = torch.tensor(total / max(1, batches), device=self.context.device)
        return float(reduce_mean(value, self.context).cpu())

    @torch.inference_mode()
    def validate_sampling(
        self,
        loader: Optional[Iterable[Dict[str, Any]]],
        seed: int,
        max_batches: int,
    ) -> Optional[Dict[str, float]]:
        """Rank-0 fixed-subset reverse diffusion validation using EMA weights."""
        barrier(self.context)
        if not self.context.is_main:
            barrier(self.context)
            return None
        if loader is None:
            raise ValueError("Main rank requires a non-distributed sampling loader")
        base_model = unwrap_model(self.model)
        base_model.eval()
        generator = _device_generator(self.context.device, seed)
        si_values, sdr_values = [], []
        with self.ema.average_parameters(base_model):
            for batch_index, raw_batch in enumerate(loader):
                if batch_index >= max_batches:
                    break
                batch = self._move(raw_batch)
                for item_index, length in enumerate(batch["lengths"].tolist()):
                    noisy = batch["noisy"][item_index, :length]
                    clean = batch["clean"][item_index, :length]
                    result = enhance_configured(self.sampler, base_model, noisy, self.config, generator)
                    row = compute_audio_metrics(result.waveform[0], clean, noisy,
                        int(self.config["data"]["sample_rate"]), **self.config["metrics"])[0]
                    if row["valid"]:
                        si_values.append(row["si_snri"])
                        sdr_values.append(row["sdri"])
        metrics = {
            "si_snri": float(sum(si_values) / len(si_values)) if si_values else float("nan"),
            "sdri": float(sum(sdr_values) / len(sdr_values)) if sdr_values else float("nan"),
            "valid_count": float(len(si_values)),
        }
        barrier(self.context)
        return metrics

    def _save(self, epoch: int, name: str) -> None:
        local_state = {"rng": capture_rng_state(),
                       "loader_rng": self.train_loader.generator.get_state()}
        rank_states = [None] * self.context.world_size
        if self.context.enabled:
            torch.distributed.all_gather_object(rank_states, local_state)
        else:
            rank_states = [local_state]
        if not self.context.is_main:
            return
        state = training_state(
            unwrap_model(self.model),
            self.ema,
            self.optimizer,
            self.scheduler,
            self.scaler,
            epoch,
            self.global_step,
            self.best_metric,
            self.config,
        )
        state["rank_states"] = rank_states
        save_checkpoint(state, str(self.checkpoint_dir / name))

    def fit(
        self,
        train_loader: Any,
        valid_loader: Any,
        sampling_valid_loader: Optional[Any],
    ) -> None:
        """Train, validate the objective each epoch, and sample at fixed intervals."""
        training = self.config["training"]
        validation = self.config["validation"]
        epochs = int(training["epochs"])
        eval_interval = int(validation.get("eval_interval", 5))
        validation_seed = int(validation.get("seed", 1234))
        max_batches = int(validation.get("max_sampling_batches", 1))
        self.train_loader = train_loader
        if self.resume_payload is not None:
            states = self.resume_payload.get("rank_states")
            if states and len(states) != self.context.world_size:
                raise ValueError("Exact epoch resume requires the same world size")
            if states:
                local = states[self.context.rank]
                restore_rng_state(local["rng"])
                train_loader.generator.set_state(local["loader_rng"].cpu())
            else:
                restore_rng_state(self.resume_payload["rng_state"])
        for epoch in range(self.start_epoch, epochs):
            started = time.time()
            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            train_stats = self.train_one_epoch(train_loader, epoch)
            val_loss = self.validate_objective(valid_loader, validation_seed)
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            sample_metrics = None
            if (epoch + 1) % eval_interval == 0 or epoch + 1 == epochs:
                sample_metrics = self.validate_sampling(
                    sampling_valid_loader, validation_seed, max_batches
                )
            improved = False
            if self.context.is_main and sample_metrics is not None:
                improved = sample_metrics["valid_count"] > 0 and sample_metrics["si_snri"] > self.best_metric
                if improved:
                    self.best_metric = sample_metrics["si_snri"]
            record: Dict[str, Any] = {
                "epoch": epoch + 1,
                "global_step": self.global_step,
                "train/loss": train_stats["loss"],
                "train/grad_norm": train_stats["grad_norm"],
                "train/lr": train_stats["lr"],
                "train/sigma_mean": train_stats["sigma_mean"],
                "train/t_mean": train_stats["t_mean"],
                "val/loss": val_loss,
                "val/" + self.objective.process.loss_name: val_loss,
                "epoch_seconds": time.time() - started,
            }
            if self.objective.process.kind != "score":
                record.pop("train/sigma_mean")
            if self.context.enabled:
                sync = [improved, self.best_metric]
                torch.distributed.broadcast_object_list(sync, src=0)
                improved, self.best_metric = sync
            if sample_metrics is not None:
                record["val/SI-SNRi"] = sample_metrics["si_snri"]
                record["val/SDRi"] = sample_metrics["sdri"]
            if self.context.is_main:
                self.jsonl.write(record)
                self.logger.info(
                    "epoch=%d loss=%.6f val_loss=%.6f grad=%.3f "
                    "lr=%.3e sigma=%.4f t=%.4f SI-SNRi=%s SDRi=%s time=%.1fs",
                    epoch + 1,
                    train_stats["loss"],
                    val_loss,
                    train_stats["grad_norm"],
                    train_stats["lr"],
                    train_stats["sigma_mean"],
                    train_stats["t_mean"],
                    "n/a" if sample_metrics is None else "{:.3f}".format(sample_metrics["si_snri"]),
                    "n/a" if sample_metrics is None else "{:.3f}".format(sample_metrics["sdri"]),
                    record["epoch_seconds"],
                )
                if self.writer is not None:
                    for key, value in record.items():
                        if isinstance(value, (int, float)):
                            self.writer.add_scalar(key, value, epoch + 1)
            self._save(epoch, "last.pt")
            if improved:
                self._save(epoch, "best.pt")
            save_every = int(training.get("save_every", 10))
            if save_every > 0 and (epoch + 1) % save_every == 0:
                self._save(epoch, "epoch_{:04d}.pt".format(epoch + 1))
            barrier(self.context)
        if self.context.is_main and self.writer is not None:
            self.writer.close()



