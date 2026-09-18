from pathlib import Path
import csv,json
import torch
from torch.utils.data import DataLoader,TensorDataset
from cdiffuse.factory import build_components
from cdiffuse.utils.ema import EMA
from cdiffuse.utils.checkpoint import save_checkpoint,load_checkpoint
from cdiffuse.diffusion.process import q_sample,combined_noise_target
from cdiffuse.diffusion.losses import masked_mse
from cdiffuse.data.audio_io import save_audio
from cdiffuse.metrics.audio_metrics import metric_row

def test_cpu_smoke(tmp_path):
    cfg={'conditioner':{'n_fft':16,'win_length':16,'hop_length':4},'diffusion':{'num_steps':2,'beta_start':1e-4,'beta_end':.01,'variance_floor':1e-12},'model':{'residual_layers':1,'residual_channels':4,'dilation_cycle_length':1,'kernel_size':3,'conditioner_channels':4},'sampler':{'seed':3}}
    model,schedule,sampler=build_components(cfg);opt=torch.optim.Adam(model.parameters(),1e-3);ema=EMA(model,.9);scaler=torch.cuda.amp.GradScaler(enabled=False);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
    tline=torch.linspace(0,1,32);clean=torch.stack([torch.sin(2*torch.pi*f*tline) for f in (2.,3.,4.,5.)]);noisy=clean+.05*torch.randn_like(clean);loader=DataLoader(TensorDataset(clean,noisy),batch_size=2)
    for x,y in loader: # exactly two CPU batches
        t=torch.randint(1,3,(2,));xt,e,_=q_sample(schedule,x,y,t);loss=masked_mse(model(xt,y,t),combined_noise_target(schedule,x,y,e,t),torch.tensor([32,32]));opt.zero_grad();loss.backward();opt.step();ema.update(model)
    ck=tmp_path/'last.pt';save_checkpoint(ck,model,ema,opt,scheduler,scaler,0,2,float(loss),cfg);restored,_,_=build_components(cfg);load_checkpoint(ck,restored,'cpu');raw=sampler.sample(restored,noisy[:1]).waveform[0];final=.8*raw+.2*noisy[0];row=metric_row(raw,final,clean[0],noisy[0]);save_audio(tmp_path/'enhanced.wav',final,16000)
    with open(tmp_path/'metrics.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerow(row)
    json.dump({'valid':row['valid'],'nfe':2},open(tmp_path/'summary.json','w'))
    assert all((tmp_path/x).exists() for x in ('last.pt','enhanced.wav','metrics.csv','summary.json'))
