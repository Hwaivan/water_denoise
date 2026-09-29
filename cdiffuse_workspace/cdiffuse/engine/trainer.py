"""CDiffuSE trainer with SGMSE-aligned logging and sampling validation."""

import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

from ..diffusion.losses import masked_mse
from ..diffusion.process import combined_noise_target, q_sample
from ..metrics.audio_metrics import compute_audio_metrics
from ..utils.checkpoint import load_checkpoint, save_checkpoint


def _device_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except TypeError:
        generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


class Trainer:
    def __init__(
        self,
        model,
        schedule,
        sampler,
        optimizer,
        scheduler,
        ema,
        config: Dict[str, Any],
        device: torch.device,
        logger,
        jsonl_logger,
        writer: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.schedule = schedule
        self.sampler = sampler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.c = config
        self.device = device
        self.logger = logger
        self.jsonl = jsonl_logger
        self.writer = writer

        training = config["training"]
        self.grad_clip = float(training.get("grad_clip_norm", 5.0))
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=bool(training.get("amp", True) and device.type == "cuda")
        )

        self.step = 0
        self.start = 0
        # SGMSE uses generative SI-SNRi for best checkpoint selection.
        self.best = -math.inf

        self.output_dir = Path(config["project"]["output_dir"])
        self.checkpoint_dir = self.output_dir / "checkpoints"

    def resume(self, path: str) -> None:
        ck = load_checkpoint(
            path,
            self.model,
            self.device,
            self.ema,
            self.optimizer,
            self.scheduler,
            self.scaler,
        )
        self.start = int(ck["epoch"]) + 1
        self.step = int(ck.get("global_step", 0))

        # Old CDiffuSE checkpoints stored validation-loss minima in best_metric.
        # Only restore it when the saved config explicitly says SI-SNRi.
        saved_config = ck.get("config", {})
        saved_validation = saved_config.get("validation", {})
        if saved_validation.get("best_metric") == "si_snri":
            self.best = float(ck.get("best_metric", -math.inf))
        else:
            self.best = -math.inf
            self.logger.warning(
                "Checkpoint predates SI-SNRi best-model selection; "
                "best_metric has been reset to -inf."
            )

        self.logger.info(
            "Resumed %s at epoch=%d global_step=%d best_SI-SNRi=%s",
            path,
            self.start,
            self.step,
            "n/a" if not math.isfinite(self.best) else f"{self.best:.3f}",
        )

    def _move(self, batch: Dict[str, Any]):
        return (
            batch["clean"].to(self.device, non_blocking=True),
            batch["noisy"].to(self.device, non_blocking=True),
            batch["lengths"].to(self.device, non_blocking=True),
        )

    def train_one_epoch(self, loader: Iterable[Dict[str, Any]]) -> Dict[str, float]:
        """Train one epoch and report the same diagnostics as SGMSE."""
        self.model.train()
        totals = {
            "loss": 0.0,
            "grad_norm": 0.0,
            "sigma_mean": 0.0,
            "t_mean": 0.0,
            "m_mean": 0.0,
            "batches": 0.0,
        }

        for batch in loader:
            clean, noisy, lengths = self._move(batch)
            batch_size = clean.shape[0]
            t = torch.randint(
                1,
                self.schedule.num_steps + 1,
                (batch_size,),
                device=self.device,
            )
            xt, epsilon, _ = q_sample(self.schedule, clean, noisy, t)
            target = combined_noise_target(
                self.schedule, clean, noisy, epsilon, t
            )

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                prediction = self.model(xt, noisy, t)
                loss = masked_mse(prediction, target, lengths)

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite CDiffuSE training loss")

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Non-finite gradient norm")

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update(self.model)
            self.step += 1

            delta_t = self.schedule.extract(self.schedule.delta, t, clean)
            m_t = self.schedule.extract(self.schedule.m, t, clean)

            totals["loss"] += float(loss.detach())
            totals["grad_norm"] += float(grad_norm.detach())
            totals["sigma_mean"] += float(
                torch.sqrt(
                    delta_t.clamp_min(self.schedule.variance_floor)
                ).mean()
            )
            totals["t_mean"] += float(t.float().mean())
            totals["m_mean"] += float(m_t.mean())
            totals["batches"] += 1.0

        count = max(1.0, totals.pop("batches"))
        stats = {name: value / count for name, value in totals.items()}
        stats["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return stats

    @torch.inference_mode()
    def validate_objective(
        self,
        loader: Iterable[Dict[str, Any]],
        seed: int,
    ) -> float:
        """Deterministic combined-noise MSE validation with fixed t/epsilon."""
        self.model.eval()
        generator = _device_generator(self.device, seed)
        total = 0.0
        count = 0

        for batch in loader:
            clean, noisy, lengths = self._move(batch)
            batch_size = clean.shape[0]
            t = torch.randint(
                1,
                self.schedule.num_steps + 1,
                (batch_size,),
                device=self.device,
                generator=generator,
            )
            epsilon = torch.randn(
                clean.shape,
                device=clean.device,
                dtype=clean.dtype,
                generator=generator,
            )
            xt, epsilon, _ = q_sample(
                self.schedule, clean, noisy, t, epsilon=epsilon
            )
            target = combined_noise_target(
                self.schedule, clean, noisy, epsilon, t
            )
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                prediction = self.model(xt, noisy, t)
                loss = masked_mse(prediction, target, lengths)

            total += float(loss)
            count += 1

        return total / max(1, count)

    @torch.inference_mode()
    def validate_sampling(
        self,
        loader: Iterable[Dict[str, Any]],
        seed: int,
        max_batches: int,
    ) -> Dict[str, float]:
        """Run EMA full reverse diffusion and compute raw SI-SNRi/SDRi."""
        ema_model = self.ema.model
        ema_model.eval()
        generator = _device_generator(self.device, seed)

        metric_cfg = self.c.get("metrics", {})
        sample_rate = int(self.c["data"]["sample_rate"])
        si_values = []
        sdr_values = []

        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break

            clean, noisy, lengths = self._move(batch)
            result = self.sampler.sample(
                ema_model,
                noisy,
                generator=generator,
                save_interval=0,
            )

            # Crop every item to its true length before SI-SNR/SDR calculation.
            for item_index in range(clean.shape[0]):
                length = int(lengths[item_index].item())
                enhanced_i = result.waveform[item_index, :length]
                clean_i = clean[item_index, :length]
                noisy_i = noisy[item_index, :length]

                row = compute_audio_metrics(
                    enhanced_i,
                    clean_i,
                    noisy_i,
                    sample_rate=sample_rate,
                    alignment_policy=metric_cfg.get("alignment_policy", "crop"),
                    eps=float(metric_cfg.get("eps", 1.0e-8)),
                    min_db=metric_cfg.get("min_db"),
                    max_db=metric_cfg.get("max_db"),
                )[0]
                if row["valid"]:
                    si_values.append(row["si_snri"])
                    sdr_values.append(row["sdri"])

        return {
            "si_snri": float(sum(si_values) / max(1, len(si_values))),
            "sdri": float(sum(sdr_values) / max(1, len(sdr_values))),
            "valid_count": float(len(si_values)),
        }

    def _save(self, epoch: int, name: str) -> None:
        save_checkpoint(
            str(self.checkpoint_dir / name),
            self.model,
            self.ema,
            self.optimizer,
            self.scheduler,
            self.scaler,
            epoch,
            self.step,
            self.best,
            self.c,
        )

    def fit(self, train_loader, valid_loader, sampling_valid_loader) -> None:
        training = self.c["training"]
        validation = self.c.get("validation", {})

        epochs = int(training["epochs"])
        eval_interval = int(validation.get("eval_interval", 5))
        validation_seed = int(validation.get("seed", 1234))
        max_batches = int(validation.get("max_sampling_batches", 2))
        save_every = int(training.get("save_every", 10))

        for epoch in range(self.start, epochs):
            started = time.time()

            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)

            train_stats = self.train_one_epoch(train_loader)
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
                    sampling_valid_loader,
                    validation_seed,
                    max_batches,
                )

            improved = False
            if sample_metrics is not None:
                improved = sample_metrics["si_snri"] > self.best
                if improved:
                    self.best = sample_metrics["si_snri"]

            record: Dict[str, Any] = {
                "epoch": epoch + 1,
                "global_step": self.step,
                "train/loss": train_stats["loss"],
                "train/grad_norm": train_stats["grad_norm"],
                "train/lr": train_stats["lr"],
                "train/sigma_mean": train_stats["sigma_mean"],
                "train/t_mean": train_stats["t_mean"],
                "train/m_mean": train_stats["m_mean"],
                "val/objective_loss": val_loss,
                "epoch_seconds": time.time() - started,
            }
            if sample_metrics is not None:
                record["val/SI-SNRi"] = sample_metrics["si_snri"]
                record["val/SDRi"] = sample_metrics["sdri"]
                record["val/valid_count"] = sample_metrics["valid_count"]

            self.jsonl.write(record)
            self.logger.info(
                "epoch=%d loss=%.6f val_loss=%.6f grad=%.3f "
                "lr=%.3e sigma=%.4f t=%.3f m=%.4f "
                "SI-SNRi=%s SDRi=%s valid=%s time=%.1fs",
                epoch + 1,
                train_stats["loss"],
                val_loss,
                train_stats["grad_norm"],
                train_stats["lr"],
                train_stats["sigma_mean"],
                train_stats["t_mean"],
                train_stats["m_mean"],
                "n/a"
                if sample_metrics is None
                else "{:.3f}".format(sample_metrics["si_snri"]),
                "n/a"
                if sample_metrics is None
                else "{:.3f}".format(sample_metrics["sdri"]),
                "n/a"
                if sample_metrics is None
                else str(int(sample_metrics["valid_count"])),
                record["epoch_seconds"],
            )

            if self.writer is not None:
                for key, value in record.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(key, value, epoch + 1)

            self._save(epoch, "last.pt")
            if improved:
                self._save(epoch, "best.pt")
            if save_every > 0 and (epoch + 1) % save_every == 0:
                self._save(epoch, "epoch_{:04d}.pt".format(epoch + 1))

        if self.writer is not None:
            self.writer.close()
