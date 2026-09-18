import math
import torch
from torch import nn
from .diffusion_embedding import DiffusionEmbedding
from .conditioner import NoisySpectrumConditioner
from .residual_block import ResidualBlock

class DiffWave(nn.Module):
    """Non-causal 1-D DiffWave. Inputs/outputs are real [B,1,L] float tensors."""
    def __init__(self,num_steps=50,residual_layers=30,residual_channels=63,dilation_cycle_length=10,kernel_size=3,conditioner_channels=128,n_fft=1024,win_length=1024,hop_length=256):
        super().__init__(); self.input=nn.Conv1d(1,residual_channels,1); self.embedding=DiffusionEmbedding(num_steps)
        self.conditioner=NoisySpectrumConditioner(conditioner_channels,n_fft,win_length,hop_length)
        self.blocks=nn.ModuleList([ResidualBlock(residual_channels,conditioner_channels,2**(i%dilation_cycle_length),kernel_size) for i in range(residual_layers)])
        self.skip=nn.Conv1d(residual_channels,residual_channels,1); self.output=nn.Conv1d(residual_channels,1,1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
    def forward(self,xt,noisy,t):
        if xt.ndim==2: xt=xt.unsqueeze(1)
        if noisy.ndim==2: noisy=noisy.unsqueeze(1)
        if xt.shape!=noisy.shape: raise ValueError("xt/noisy must share [B,1,L]")
        x=torch.relu(self.input(xt)); emb=self.embedding(t); cond=self.conditioner(noisy,xt.shape[-1]); skips=[]
        for block in self.blocks: x,s=block(x,emb,cond); skips.append(s)
        return self.output(torch.relu(self.skip(torch.stack(skips).sum(0)/math.sqrt(len(skips)))))
