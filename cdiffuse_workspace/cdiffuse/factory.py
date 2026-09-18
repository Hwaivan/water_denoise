from .diffusion import CDiffuSESchedule,CDiffuSESampler
from .models import DiffWave
def build_components(c):
    d=c['diffusion'];m=c['model'];q=c['conditioner']
    schedule=CDiffuSESchedule(d['num_steps'],d['beta_start'],d['beta_end'],d.get('variance_floor',1e-12))
    model=DiffWave(d['num_steps'],m['residual_layers'],m['residual_channels'],m['dilation_cycle_length'],m['kernel_size'],m['conditioner_channels'],q['n_fft'],q['win_length'],q['hop_length'])
    return model,schedule,CDiffuSESampler(schedule,c['sampler'].get('seed',1234))
