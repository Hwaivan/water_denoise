import importlib.util
from pathlib import Path
import torch
from dit.processes.score_sde import ScoreProcess
from dit.samplers.score_pc import ScorePC
from dit.representations.stft import complex_to_channels, channels_to_complex


def process():
    return ScoreProcess("circular_complex_unit_energy")


def test_ouve_and_target_match_local_reference():
    path = Path(__file__).resolve().parents[2] / "sgmse_workspace/sgmse/diffusion/ouve.py"
    if path.exists():
        spec = importlib.util.spec_from_file_location("legacy_ouve", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        old = module.OUVESDE()
    else:
        old = None
    p = process()
    t = torch.tensor([.03,.2,.5,1.])
    x, y = torch.randn(4,2,9,7), torch.randn(4,2,9,7)
    z = p.noise(x, torch.Generator().manual_seed(4))
    pair = p.sample_training_pair(x,y,time=t,noise=z)
    std = p.sde.std(t)[:,None,None,None]
    torch.testing.assert_close((pair.state-p.sde.mean(x,y,t))/std,z)
    torch.testing.assert_close(pair.target, -z/std)
    # Fixed numeric values independent of the implementation under test.
    import math
    expected = .05**2*(10**(2*t)-torch.exp(-3*t))/(1.5+math.log(10))
    torch.testing.assert_close(p.sde.variance(t),expected)
    if old is not None:
        torch.testing.assert_close(p.sde.variance(t),old.variance(t),rtol=0,atol=0)
        torch.testing.assert_close(p.sde.diffusion(t),old.diffusion(t),rtol=0,atol=0)
        torch.testing.assert_close(channels_to_complex(p.sde.mean(x,y,t)),
                                   old.mean(channels_to_complex(x), channels_to_complex(y),t))
        legacy_noise=module.complex_gaussian((4,9,7),torch.device('cpu'),generator=torch.Generator().manual_seed(4))
        torch.testing.assert_close(z,complex_to_channels(legacy_noise),rtol=0,atol=0)


def test_noise_energy():
    z = process().noise(torch.empty(8,2,256,128),torch.Generator().manual_seed(123))
    assert abs(z.square().sum(1).mean().item()-1)<.01


def test_pc_nfe_and_reproducibility():
    class Zero(torch.nn.Module):
        def forward(self,state,condition,time):
            return torch.zeros_like(state)
    p = process()
    s = ScorePC(p,num_steps=3)
    y = torch.zeros(2,2,9,7)
    a,nfe = s.sample(Zero(),y,torch.Generator().manual_seed(42))
    b,_ = s.sample(Zero(),y,torch.Generator().manual_seed(42))
    assert nfe == 6 and a.shape == y.shape and torch.isfinite(a).all()
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_pc_matches_legacy_sampler():
    import sys
    import pytest
    legacy = Path(__file__).resolve().parents[2] / "sgmse_workspace"
    if not legacy.exists():
        pytest.skip("Optional sibling legacy sampler unavailable in standalone distribution")
    sys.path.insert(0, str(legacy))
    from sgmse.diffusion.ouve import OUVESDE
    from sgmse.diffusion.sampler import PredictorCorrectorSampler
    from sgmse.utils.stft import ComplexSTFTTransform
    class Old(torch.nn.Module):
        def forward(self, x, t):
            return -.2*x[:,:2] + .1*x[:,2:]
    class New(torch.nn.Module):
        def forward(self,state,condition,time):
            return -.2*state+.1*condition
    y = torch.randn(2,2,9,7)
    old = PredictorCorrectorSampler(OUVESDE(),ComplexSTFTTransform(64,48,16),num_steps=3)
    expected,nfe_old=old.sample_spectrum(Old(),channels_to_complex(y),torch.Generator().manual_seed(7))
    actual,nfe_new=ScorePC(process(),num_steps=3).sample(New(),y,torch.Generator().manual_seed(7))
    torch.testing.assert_close(actual,complex_to_channels(expected),rtol=1e-5,atol=1e-6)
    assert nfe_old == nfe_new == 6
