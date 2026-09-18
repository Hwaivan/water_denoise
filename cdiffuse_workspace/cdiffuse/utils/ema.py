import copy
import torch
class EMA:
    def __init__(self,model,decay=.999):self.decay=decay;self.model=copy.deepcopy(model).eval();self.model.requires_grad_(False)
    @torch.no_grad()
    def update(self,model):
        for e,p in zip(self.model.parameters(),model.parameters()):e.mul_(self.decay).add_(p,alpha=1-self.decay)
    def state_dict(self):return self.model.state_dict()
    def load_state_dict(self,s):self.model.load_state_dict(s)
