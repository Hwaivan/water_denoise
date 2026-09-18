import math
import torch
from torch import nn

class DiffusionEmbedding(nn.Module):
    def __init__(self,max_steps,dim=128):
        super().__init__(); half=dim//2
        steps=torch.arange(max_steps+1).float()[:,None]
        freq=torch.exp(torch.arange(half).float()*(-math.log(10000)/max(1,half-1)))[None]
        self.register_buffer("table",torch.cat((torch.sin(steps*freq),torch.cos(steps*freq)),1),persistent=False)
        self.proj=nn.Sequential(nn.Linear(dim,dim*4),nn.SiLU(),nn.Linear(dim*4,dim))
    def forward(self,t): return self.proj(self.table[t.long()])
