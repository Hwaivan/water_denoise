import torch
def _sdr(e,r,eps=1e-8):return 10*torch.log10((r.square().sum()+eps)/((e-r).square().sum()+eps))
def _si(e,r,eps=1e-8):
    e=e-e.mean();r=r-r.mean();target=(e*r).sum()/r.square().sum().clamp_min(eps)*r;return 10*torch.log10((target.square().sum()+eps)/((e-target).square().sum()+eps))
def metric_row(raw,final,clean,noisy):
    if not all(torch.isfinite(x).all() for x in (raw,final,clean,noisy)) or clean.square().sum()<=1e-8:return {'valid':False,'error':'non_finite_or_silent'}
    ins,rs,fs=_sdr(noisy,clean),_sdr(raw,clean),_sdr(final,clean);ini,ri,fi=_si(noisy,clean),_si(raw,clean),_si(final,clean)
    return {k:float(v) for k,v in {'input_sdr':ins,'raw_output_sdr':rs,'final_output_sdr':fs,'raw_sdri':rs-ins,'final_sdri':fs-ins,'input_si_snr':ini,'raw_output_si_snr':ri,'final_output_si_snr':fi,'raw_si_snri':ri-ini,'final_si_snri':fi-ini}.items()}|{'valid':True,'error':''}
