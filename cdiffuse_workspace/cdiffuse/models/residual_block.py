import math
import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self,channels,conditioner_channels,dilation,kernel_size=3,embedding_dim=128):
        super().__init__(); pad=dilation*(kernel_size-1)//2
        self.diffusion=nn.Linear(embedding_dim,channels)
        self.condition=nn.Conv1d(conditioner_channels,2*channels,1)
        self.dilated=nn.Conv1d(channels,2*channels,kernel_size,padding=pad,dilation=dilation)
        self.output=nn.Conv1d(channels,2*channels,1)
    def forward(self,x,diffusion_embedding,conditioner):
        h=x+self.diffusion(diffusion_embedding).unsqueeze(-1)
        gate,fil=(self.dilated(h)+self.condition(conditioner)).chunk(2,1)
        h=torch.sigmoid(gate)*torch.tanh(fil)
        residual,skip=self.output(h).chunk(2,1)
        return (x+residual)/math.sqrt(2.0),skip
