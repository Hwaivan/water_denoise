"""Euler-Maruyama/annealed Langevin, matching the local SGMSE equations."""
import torch
from dit.processes.base import broadcast


class ScorePC:
    def __init__(self, process, num_steps=30, corrector="annealed_langevin", corrector_steps=1,
                 corrector_step_size=.5, predictor="euler_maruyama"):
        if num_steps < 1 or corrector_steps < 0 or corrector_step_size < 0:
            raise ValueError("Invalid PC steps")
        if predictor != "euler_maruyama" or corrector not in ("annealed_langevin", "none"):
            raise ValueError("Unsupported predictor/corrector")
        self.process, self.num_steps = process, num_steps
        self.corrector_steps = corrector_steps if corrector != "none" else 0
        self.corrector_step_size = corrector_step_size

    @torch.inference_mode()
    def sample(self, model, condition, generator=None):
        p, sde = self.process, self.process.sde
        time = condition.new_full((condition.shape[0],), sde.T)
        state = condition + broadcast(sde.std(time), condition) * p.noise(condition, generator)
        grid = torch.linspace(sde.T, sde.t_eps, self.num_steps + 1, device=condition.device)
        nfe = 0
        for index in range(self.num_steps):
            time = grid[index].expand(condition.shape[0])
            for _ in range(self.corrector_steps):
                score = model(state=state, condition=condition, time=time).float()
                nfe += 1
                step = self.corrector_step_size * broadcast(sde.std(time).square(), state)
                state = state + step * score + (2 * step).sqrt() * p.noise(state, generator)
            score = model(state=state, condition=condition, time=time).float()
            nfe += 1
            delta = grid[index+1] - grid[index]
            state = state + sde.reverse_drift(state, condition, time, score) * delta
            if index < self.num_steps - 1:
                state = state + broadcast(sde.diffusion(time), state) * (-delta).sqrt() * p.noise(state, generator)
            if not torch.isfinite(state).all():
                raise FloatingPointError("Non-finite PC state")
        return state, nfe
