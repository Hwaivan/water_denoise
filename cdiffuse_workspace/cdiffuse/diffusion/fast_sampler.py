"""DiffWave-style effective-alpha mapping for experimental fast inference."""
import torch

DEFAULT_FAST_SCHEDULE=(0.0001,0.001,0.01,0.05,0.2,0.35)

def map_inference_steps(train_alpha_bar, inference_betas=DEFAULT_FAST_SCHEDULE):
    """Map inference cumulative alphas to fractional training indices.

    This implements DiffWave's log-alpha interpolation and does not select an
    equally spaced subset. Sampling remains opt-in because skipped conditional
    posterior coefficients require separate derivation/validation.
    """
    betas=torch.as_tensor(inference_betas,dtype=torch.float64)
    if not ((betas>0)&(betas<1)).all(): raise ValueError("fast betas must be in (0,1)")
    infer_ab=torch.cumprod(1-betas,0)
    train=train_alpha_bar[1:].double()
    mapped=[]
    for value in infer_ab:
        hits=torch.nonzero((train[:-1]>=value)&(value>=train[1:]),as_tuple=False)
        if hits.numel()==0:
            mapped.append(torch.argmin((train-value).abs()).double())
        else:
            i=int(hits[0]); w=(torch.log(train[i])-torch.log(value))/(torch.log(train[i])-torch.log(train[i+1]))
            mapped.append(torch.tensor(i,dtype=torch.float64)+w)
    result=torch.stack(mapped)
    if not torch.isfinite(result).all(): raise ValueError("invalid fast-step mapping")
    return result
