"""Write inspectable, deterministic one-step diagnostics; never a long training run."""
import copy
import json
from pathlib import Path
import torch
from dit.factory import build_components
from dit.utils.config import load_config
from dit.utils.seed import seed_everything


def main():
    torch.set_num_threads(1)
    seed_everything(42)
    base=load_config(str(Path(__file__).parent/'configs/dit_tiny.yaml'))
    rows=[]
    for representation in ('complex_ri','magnitude'):
        for kind in ('score','ddpm','flow'):
            seed_everything(42)
            c=copy.deepcopy(base)
            c['representation']['type']=representation
            c['process']['type']=kind
            c['process']['noise_convention']='circular_complex_unit_energy' if representation=='complex_ri' else 'real_unit_variance'
            if kind!='score':
                c['process']['ddpm']['num_steps']=3
                c['sampler']=dict(name='ddpm' if kind=='ddpm' else 'euler',num_steps=3)
            model,rep,process,objective,sampler=build_components(c)
            wave=.2*torch.sin(torch.arange(256)*.31)[None]
            noisy=wave+.07*torch.randn_like(wave)
            loss=objective(model,wave,noisy)['total']
            opt=torch.optim.Adam(model.parameters(),lr=1e-4)
            loss.backward()
            opt.step()
            model.eval()
            state=rep.encode(noisy)
            output=model(state=state,condition=state,time=torch.tensor([.5]))
            sample=sampler.sample_waveform(model,noisy,generator=torch.Generator().manual_seed(1234))
            rows.append(dict(process=kind,representation=representation,input_shape=list(state.shape),
                             concatenated_shape=[state.shape[0],state.shape[1]*2,*state.shape[2:]],
                             output_shape=list(output.shape),**model.geometry(*state.shape[-2:]),
                             training_geometry=model.geometry(state.shape[-2],c['compression']['crop_frames']),
                             one_step_loss=loss.item(),waveform_shape=list(sample.waveform.shape),
                             nfe=sample.nfe,peak_gpu_memory=None))
    small=load_config(str(Path(__file__).parent/'configs/dit_score_complex_small.yaml'))
    model,_,_,_,_=build_components(small)
    report=dict(torch_version=torch.__version__,device='cpu',cuda_available=torch.cuda.is_available(),
                seed=42,tiny=rows,small_geometry=model.geometry(257,128))
    path=Path(__file__).parent/'SMOKE_RESULTS.json'
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
