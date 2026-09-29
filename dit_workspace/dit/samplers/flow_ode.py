"""Fixed-step Euler and Heun on [0,1], with exact network-call accounting."""
import torch


class FlowODE:
    def __init__(self, process, num_steps=30, solver="euler"):
        if solver not in ("euler","heun") or num_steps < 1:
            raise ValueError("Invalid flow solver or num_steps")
        self.process, self.num_steps, self.solver = process, int(num_steps), solver

    @torch.inference_mode()
    def sample(self, model, condition, generator=None):
        state = self.process.noise(condition,generator)
        dt, nfe = 1/self.num_steps, 0
        for index in range(self.num_steps):
            time = condition.new_full((condition.shape[0],),index*dt)
            velocity = model(state=state,condition=condition,time=time).float()
            nfe += 1
            prediction = state + dt*velocity
            if self.solver == "heun":
                next_time = condition.new_full((condition.shape[0],),(index+1)*dt)
                next_velocity = model(state=prediction,condition=condition,time=next_time).float()
                nfe += 1
                state = state + dt*.5*(velocity+next_velocity)
            else:
                state = prediction
            if not torch.isfinite(state).all():
                raise FloatingPointError("Non-finite flow state")
        return state,nfe
