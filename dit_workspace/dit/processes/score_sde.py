"""Explicit local OUVE baseline, operating on real representation channels."""
import torch
from .base import Process, TrainingPair, broadcast
from .ouve import OUVESDE


class ScoreProcess(Process):
    kind, loss_name = "score", "score_loss"

    def __init__(self, noise_convention, ouve_variant="water_denoise", sde="ouve", **kwargs):
        super().__init__(noise_convention)
        if ouve_variant != "water_denoise" or sde != "ouve":
            raise NotImplementedError("Only the explicit water_denoise OUVE baseline is implemented")
        self.sde = OUVESDE(**kwargs)

    def sample_training_pair(self, clean, condition, generator=None, deterministic=False, time=None, noise=None):
        if clean.shape != condition.shape:
            raise ValueError("Clean/condition shapes differ")
        if time is None:
            u = torch.full((clean.shape[0],), .5, device=clean.device) if deterministic else torch.rand(
                clean.shape[0], device=clean.device, generator=generator)
            time = self.sde.t_eps + (self.sde.T - self.sde.t_eps) * u
        noise = self.noise(clean, generator) if noise is None else noise
        sigma = self.sde.std(time)
        state = self.sde.mean(clean, condition, time) + broadcast(sigma, clean) * noise
        return TrainingPair(state, time, -noise / broadcast(sigma, clean).clamp_min(1e-8),
                            {"sigma_mean": sigma.mean(), "t_mean": time.float().mean()})
