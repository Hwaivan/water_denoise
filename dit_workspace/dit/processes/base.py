"""Backbone-independent training pair contract."""
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch
from torch import nn
from dit.utils.noise import sample_noise


def broadcast(value, reference):
    return value.reshape(value.shape[0], *([1] * (reference.ndim - 1)))


@dataclass
class TrainingPair:
    state: torch.Tensor
    time: torch.Tensor
    target: torch.Tensor
    diagnostics: dict


class Process(nn.Module, ABC):
    def __init__(self, noise_convention):
        super().__init__()
        if noise_convention not in ("circular_complex_unit_energy", "real_unit_variance"):
            raise ValueError("Unsupported noise convention")
        self.noise_convention = noise_convention

    def noise(self, reference, generator=None):
        return sample_noise(reference, self.noise_convention, generator)

    @abstractmethod
    def sample_training_pair(self, clean, condition, generator=None, deterministic=False, **kwargs):
        raise NotImplementedError
