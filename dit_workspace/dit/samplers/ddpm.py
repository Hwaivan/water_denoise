"""Full ancestral DDPM reverse chain; no silent schedule subsampling."""
import torch


class DDPMSampler:
    def __init__(self, process, num_steps):
        if num_steps != process.num_steps:
            raise ValueError("Ancestral DDPM num_steps must equal training schedule; DDIM is not implemented")
        self.process, self.num_steps = process, int(num_steps)

    @torch.inference_mode()
    def sample(self, model, condition, generator=None):
        state = self.process.noise(condition,generator)
        for index in reversed(range(self.num_steps)):
            time = torch.full((condition.shape[0],),index,device=condition.device,dtype=torch.long)
            epsilon = model(state=state,condition=condition,time=time).float()
            noise = self.process.noise(state,generator) if index else torch.zeros_like(state)
            state = self.process.reverse_step(state,time,epsilon,noise)
            if not torch.isfinite(state).all():
                raise FloatingPointError("Non-finite DDPM state")
        return state, self.num_steps
