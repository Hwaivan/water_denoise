"""Euler and Heun ODE flow on [0,1], with optional Mel condition."""
import torch


class FlowODE:
    def __init__(self, process, num_steps=30, solver="euler"):
        if solver not in ("euler", "heun") or num_steps < 1:
            raise ValueError("Invalid flow solver or num_steps")
        self.process, self.num_steps, self.solver = process, int(num_steps), solver

    @torch.inference_mode()
    def sample(self, model, condition, generator=None, aux_condition=None):
        state = self.process.noise(condition, generator)
        dt, nfe = 1 / self.num_steps, 0
        for index in range(self.num_steps):
            time = condition.new_full((condition.shape[0],), index * dt)
            kwargs = {"state": state, "condition": condition, "time": time}
            if aux_condition is not None:
                kwargs["aux_condition"] = aux_condition
            velocity = model(**kwargs).float()
            nfe += 1
            prediction = state + dt * velocity
            if self.solver == "heun":
                next_time = condition.new_full((condition.shape[0],), (index + 1) * dt)
                kwargs = {"state": prediction, "condition": condition, "time": next_time}
                if aux_condition is not None:
                    kwargs["aux_condition"] = aux_condition
                next_velocity = model(**kwargs).float()
                nfe += 1
                state = state + dt * .5 * (velocity + next_velocity)
            else:
                state = prediction
            if not torch.isfinite(state).all():
                raise FloatingPointError("Non-finite flow state")
        return state, nfe
