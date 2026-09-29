"""Conditional rectified flow with an observation-independent Gaussian source."""
import torch
from .base import Process, TrainingPair, broadcast


class FlowProcess(Process):
    kind, loss_name = "flow", "flow_loss"

    def __init__(self, noise_convention, source="gaussian"):
        super().__init__(noise_convention)
        if source != "gaussian":
            raise NotImplementedError("Noisy-start/centered sources and stochastic interpolants are reserved")

    def sample_training_pair(self, clean, condition, generator=None, deterministic=False, time=None, noise=None):
        if clean.shape != condition.shape:
            raise ValueError("Clean/condition shapes differ")
        if time is None:
            time = torch.full((clean.shape[0],),.5,device=clean.device) if deterministic else torch.rand(
                clean.shape[0],device=clean.device,generator=generator)
        source = self.noise(clean,generator) if noise is None else noise
        t = broadcast(time,clean)
        return TrainingPair((1-t)*source+t*clean,time,clean-source,{"t_mean":time.mean()})
