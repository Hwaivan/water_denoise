"""Full conditional DDPM sampler; fast mapping is isolated in fast_sampler.py."""
from dataclasses import dataclass
import torch
from .schedule import CDiffuSESchedule

@dataclass
class SampleResult:
    waveform: torch.Tensor
    nfe: int
    states: list

class CDiffuSESampler:
    def __init__(self, schedule, seed=1234): self.schedule, self.seed = schedule, int(seed)
    @torch.inference_mode()
    def sample(self, model, noisy, generator=None, save_interval=0):
        y = noisy.unsqueeze(1) if noisy.ndim == 2 else noisy
        if y.ndim != 3 or y.shape[1] != 1 or not torch.isfinite(y).all(): raise ValueError("noisy must be finite [B,1,L]")
        if generator is None:
            generator = torch.Generator(device=y.device).manual_seed(self.seed)
        abt = self.schedule.alpha_bar[-1].to(y).reshape(1,1,1)
        dt = self.schedule.delta[-1].clamp_min(self.schedule.variance_floor).to(y).reshape(1,1,1)
        x = torch.sqrt(abt)*y + torch.sqrt(dt)*torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=generator)
        states=[]
        r=self.schedule.reverse
        for step in range(self.schedule.num_steps, 0, -1):
            t=torch.full((y.shape[0],), step, device=y.device, dtype=torch.long)
            eps=model(x,y,t)
            cx=self.schedule.extract(r.c_x,t,x); cy=self.schedule.extract(r.c_y,t,x); ce=self.schedule.extract(r.c_epsilon,t,x)
            mean=cx*x+cy*y-ce*eps
            if step > 1:
                var=self.schedule.extract(r.variance,t,x).clamp_min(self.schedule.variance_floor)
                x=mean+torch.sqrt(var)*torch.randn(x.shape,device=x.device,dtype=x.dtype,generator=generator)
            else: x=mean
            if not torch.isfinite(x).all(): raise FloatingPointError(f"non-finite sampler state at t={step}, shape={tuple(x.shape)}, device={x.device}, dtype={x.dtype}")
            if save_interval and (step % save_interval == 0 or step == 1): states.append(x.detach().cpu())
        return SampleResult(x.squeeze(1),self.schedule.num_steps,states)
