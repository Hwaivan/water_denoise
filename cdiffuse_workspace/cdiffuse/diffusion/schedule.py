"""Discrete CDiffuSE coefficients, calculated in float64 for stability."""
from dataclasses import dataclass
import torch


@dataclass
class ReverseCoefficients:
    c_x: torch.Tensor
    c_y: torch.Tensor
    c_epsilon: torch.Tensor
    variance: torch.Tensor
    transition_variance: torch.Tensor


class CDiffuSESchedule:
    """Paper schedule with every tensor indexed from 0 through T.

    Public coefficient arrays are CPU float64 ``[T+1]`` tensors. State zero is
    exact: alpha_bar[0]=1, m[0]=delta[0]=0. Values gathered for a waveform are
    moved explicitly to that waveform's dtype/device by :meth:`extract`.
    """
    def __init__(self, num_steps=50, beta_start=1e-4, beta_end=0.035, variance_floor=1e-12):
        self.num_steps = int(num_steps)
        self.variance_floor = float(variance_floor)
        if self.num_steps < 1 or not (0 < beta_start <= beta_end < 1):
            raise ValueError("invalid linear beta schedule")
        beta = torch.linspace(beta_start, beta_end, self.num_steps, dtype=torch.float64)
        alpha = 1.0 - beta
        alpha_bar = torch.cat((torch.ones(1, dtype=torch.float64), alpha.cumprod(0)))
        beta = torch.cat((torch.zeros(1, dtype=torch.float64), beta))
        alpha = torch.cat((torch.ones(1, dtype=torch.float64), alpha))
        m = (1.0 - alpha_bar) / torch.sqrt(alpha_bar)
        delta = (1.0 - alpha_bar) - m.square() * alpha_bar
        m[0] = 0.0
        delta[0] = 0.0
        self.beta, self.alpha, self.alpha_bar, self.m, self.delta = beta, alpha, alpha_bar, m, delta
        self._validate()
        self.reverse = self._build_reverse()

    def _validate(self):
        values = (self.beta, self.alpha, self.alpha_bar, self.m, self.delta)
        if not all(torch.isfinite(v).all() for v in values):
            raise ValueError("schedule contains NaN/Inf")
        if not (self.beta[1:] > 0).all() or not ((self.alpha[1:] > 0) & (self.alpha[1:] < 1)).all():
            raise ValueError("beta/alpha outside valid range")
        if not (self.alpha_bar[1:] < self.alpha_bar[:-1]).all():
            raise ValueError("alpha_bar must strictly decrease")
        if self.delta.min() < -self.variance_floor:
            raise ValueError("negative forward variance: {:.6g}".format(float(self.delta.min())))
        self.delta = self.delta.clamp_min(0.0)

    def _build_reverse(self):
        # All operands below are [T], corresponding to t=1,...,T.
        mt, mp = self.m[1:], self.m[:-1]
        dt, dp = self.delta[1:], self.delta[:-1]
        at, abp = self.alpha[1:], self.alpha_bar[:-1]
        ratio = (1.0 - mt) / (1.0 - mp)  # m_0 is explicit, so t=1 is safe.
        transition = dt - ratio.square() * at * dp
        if transition.min() < -self.variance_floor:
            raise ValueError("negative reverse transition variance")
        transition = transition.clamp_min(0.0)
        safe_dt = dt.clamp_min(self.variance_floor)
        c_x = ratio * dp / safe_dt * torch.sqrt(at) + (1-mp) * transition / safe_dt / torch.sqrt(at)
        c_y = (mp*dt - mt*ratio*at*dp) * torch.sqrt(abp) / safe_dt
        c_e = (1-mp) * transition / safe_dt * torch.sqrt(1-self.alpha_bar[1:]) / torch.sqrt(at)
        variance = (transition * dp / safe_dt).clamp_min(0.0)
        # At t=1 the posterior is deterministic and its analytic mean is x0.
        variance[0] = 0.0
        arrays = [torch.cat((torch.zeros(1, dtype=torch.float64), v)) for v in (c_x,c_y,c_e,variance,transition)]
        if not all(torch.isfinite(v).all() for v in arrays):
            raise ValueError("reverse coefficients contain NaN/Inf")
        return ReverseCoefficients(*arrays)

    @staticmethod
    def extract(array, t, like):
        """Gather ``array[t]`` and reshape to broadcast with ``like [B,1,L]``."""
        if t.ndim != 1 or t.shape[0] != like.shape[0]:
            raise ValueError("t must be [B] matching waveform batch")
        out = array.to(device=like.device)[t.long()].to(dtype=like.dtype)
        return out.view(t.shape[0], *([1] * (like.ndim - 1)))
