import copy
from pathlib import Path
import pytest
import torch
import yaml
from dit.factory import build_components,build_process,build_sampler,build_objective
from dit.processes.ddpm import DDPMProcess
from dit.processes.flow_matching import FlowProcess
from dit.samplers.ddpm import DDPMSampler
from dit.samplers.flow_ode import FlowODE


@pytest.mark.parametrize("convention,channels", [("real_unit_variance",1),("circular_complex_unit_energy",2)])
def test_ddpm_forward_target_and_reverse(convention,channels):
    p = DDPMProcess(convention,num_steps=5,beta_start=.01,beta_end=.1)
    x,z,y = [torch.randn(2,channels,9,7) for _ in range(3)]
    t = torch.tensor([0,4])
    pair = p.sample_training_pair(x,y,time=t,noise=z)
    ab = p.alpha_bar[t,None,None,None]
    torch.testing.assert_close(pair.state,ab.sqrt()*x+(1-ab).sqrt()*z)
    torch.testing.assert_close(pair.target,z,atol=0,rtol=0)
    torch.testing.assert_close(pair.state,p.sample_training_pair(x,y+10,time=t,noise=z).state)
    result = p.reverse_step(pair.state,t,z,torch.randn_like(z))
    torch.testing.assert_close(result[0],x[0],atol=1e-5,rtol=1e-5)
    assert result.shape == x.shape and torch.isfinite(result).all()
    beta=p.betas[4]
    expected_variance=beta*(1-p.alpha_bar[3])/(1-p.alpha_bar[4])
    torch.testing.assert_close(p.posterior_variance[4],expected_variance)
    with pytest.raises(NotImplementedError):
        DDPMProcess(convention,prediction_type="v")
    with pytest.raises(ValueError):
        DDPMSampler(p,num_steps=2)


def test_flow_endpoints_and_velocity():
    p = FlowProcess("real_unit_variance")
    x,z,y = [torch.randn(2,1,9,7) for _ in range(3)]
    for value,expected in [(0.,z),(1.,x),(.3,.7*z+.3*x)]:
        pair=p.sample_training_pair(x,y,time=torch.full((2,),value),noise=z)
        torch.testing.assert_close(pair.state,expected)
        torch.testing.assert_close(pair.target,x-z)
        torch.testing.assert_close(pair.state,p.sample_training_pair(x,y+7,time=pair.time,noise=z).state)


@pytest.mark.parametrize("solver,multiplier", [("euler",1),("heun",2)])
def test_ode_constant_velocity_and_nfe(solver,multiplier):
    p=FlowProcess("real_unit_variance")
    y=torch.ones(2,1,9,7)
    initial=p.noise(y,torch.Generator().manual_seed(8))
    class Velocity(torch.nn.Module):
        def forward(self,state,condition,time):
            return condition*3
    result,nfe=FlowODE(p,4,solver).sample(Velocity(),y,torch.Generator().manual_seed(8))
    torch.testing.assert_close(result,initial+3)
    assert nfe==4*multiplier


@pytest.mark.parametrize("representation", ["complex_ri","magnitude"])
def test_same_backbone_across_all_processes(representation):
    config=yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/dit_tiny.yaml").read_text())
    config["representation"]["type"]=representation
    if representation=="magnitude":config["process"]["noise_convention"]="real_unit_variance"
    model,rep,_,_,_=build_components(config)
    clean,noisy=torch.randn(2,256),torch.randn(2,256)
    x,y=rep.encode(clean),rep.encode(noisy)
    model_identity=id(model)
    for kind in ("score","ddpm","flow"):
        c=copy.deepcopy(config)
        c["process"]["type"]=kind
        if kind!="score":
            c["process"]["ddpm"]["num_steps"]=3
            c["sampler"]=dict(name="ddpm" if kind=="ddpm" else "euler",num_steps=3)
        process=build_process(c,rep)
        pair=process.sample_training_pair(x,y)
        assert pair.state.shape==pair.target.shape==y.shape
        assert model(state=pair.state,condition=y,time=pair.time).shape==x.shape
        objective=build_objective(c,process,rep)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        optimizer.zero_grad()
        loss=objective(model,clean,noisy)["total"]
        loss.backward()
        assert torch.isfinite(loss) and model.final_layer.projection.weight.grad.norm()>0
        optimizer.step()
        sample=build_sampler(c,process,rep).sample_waveform(model,noisy)
        assert sample.waveform.shape==noisy.shape and torch.isfinite(sample.waveform).all()
        assert sample.nfe== (4 if kind=="score" else 3)
        assert id(model)==model_identity
