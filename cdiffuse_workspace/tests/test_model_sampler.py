import torch
from torch import nn
from cdiffuse.models import DiffWave
from cdiffuse.diffusion import CDiffuSESchedule,CDiffuSESampler
from cdiffuse.diffusion.fast_sampler import map_inference_steps
def test_model_shape():
    m=DiffWave(4,2,8,2,3,8,16,16,4);x=torch.randn(2,1,64);assert m(x,x,torch.tensor([1,4])).shape==x.shape
class Zero(nn.Module):
    def forward(self,x,y,t):return torch.zeros_like(x)
def test_sampler_seed_and_finite():
    s=CDiffuSESchedule(4,1e-4,.01);p=CDiffuSESampler(s,7);y=torch.randn(2,32);a=p.sample(Zero(),y).waveform;b=p.sample(Zero(),y).waveform;assert a.shape==y.shape and torch.isfinite(a).all() and torch.equal(a,b)
def test_fast_mapping():
    s=CDiffuSESchedule(50,1e-4,.035);m=map_inference_steps(s.alpha_bar);assert m.shape==(6,) and torch.isfinite(m).all()
