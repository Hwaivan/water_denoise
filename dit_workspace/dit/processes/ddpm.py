"""Conditional epsilon DDPM: y is not part of the forward perturbation."""
import math
import torch
from .base import Process, TrainingPair, broadcast


class DDPMProcess(Process):
    kind, loss_name = "ddpm", "noise_loss"

    def __init__(self, noise_convention, num_steps=1000, beta_schedule="linear",
                 beta_start=1e-4, beta_end=.02, prediction_type="epsilon"):
        super().__init__(noise_convention)
        if prediction_type != "epsilon":
            raise NotImplementedError("Only epsilon prediction is implemented (x0/v reserved)")
        if num_steps < 2 or not 0 < beta_start <= beta_end < 1:
            raise ValueError("Invalid DDPM step count or beta endpoints")
        self.num_steps = int(num_steps)
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.num_steps, dtype=torch.float64)
        elif beta_schedule == "cosine":
            t = torch.linspace(0,1,self.num_steps+1,dtype=torch.float64)
            cumulative = torch.cos((t+.008)/1.008 * math.pi/2).square()
            cumulative = cumulative/cumulative[0]
            betas = (1-cumulative[1:]/cumulative[:-1]).clamp(max=.999)
        else:
            raise ValueError("beta_schedule must be linear or cosine")
        alphas = 1-betas
        alpha_bar = alphas.cumprod(0)
        previous = torch.cat([torch.ones(1,dtype=torch.float64),alpha_bar[:-1]])
        for name, value in {"betas":betas, "alphas":alphas, "alpha_bar":alpha_bar,
                            "posterior_variance":betas*(1-previous)/(1-alpha_bar)}.items():
            self.register_buffer(name,value.float())

    def extract(self, name, time, reference):
        values = getattr(self,name).to(reference.device)[time.long()]
        return broadcast(values,reference)

    def q_sample(self, clean, time, noise):
        alpha_bar = self.extract("alpha_bar",time,clean)
        return alpha_bar.sqrt()*clean + (1-alpha_bar).sqrt()*noise

    def sample_training_pair(self, clean, condition, generator=None, deterministic=False, time=None, noise=None):
        if clean.shape != condition.shape:
            raise ValueError("Clean/condition shapes differ")
        if time is None:
            time = torch.full((clean.shape[0],),self.num_steps//2,device=clean.device,dtype=torch.long) if deterministic else torch.randint(
                self.num_steps,(clean.shape[0],),device=clean.device,generator=generator)
        noise = self.noise(clean,generator) if noise is None else noise
        return TrainingPair(self.q_sample(clean,time,noise),time,noise,{"t_mean":time.float().mean()})

    def reverse_step(self, state, time, epsilon, noise):
        beta = self.extract("betas",time,state)
        alpha = self.extract("alphas",time,state)
        alpha_bar = self.extract("alpha_bar",time,state)
        mean = (state-beta/(1-alpha_bar).sqrt()*epsilon)/alpha.sqrt()
        variance = self.extract("posterior_variance",time,state)
        return mean + broadcast((time>0).to(state.dtype),state)*variance.clamp_min(0).sqrt()*noise
