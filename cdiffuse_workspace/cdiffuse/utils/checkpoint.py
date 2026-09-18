import random
from pathlib import Path
import numpy as np
import torch
def rng_state():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
def save_checkpoint(path,model,ema,optimizer,scheduler,scaler,epoch,global_step,best_metric,config):
    Path(path).parent.mkdir(parents=True,exist_ok=True);torch.save({'model':model.state_dict(),'ema_model':ema.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'global_step':global_step,'best_metric':best_metric,'config':config,'rng_state':rng_state()},path)
def load_checkpoint(path,model,device,ema=None,optimizer=None,scheduler=None,scaler=None):
    ck=torch.load(path,map_location=device,weights_only=False);model.load_state_dict(ck['model'])
    if ema is not None:ema.load_state_dict(ck['ema_model'])
    for obj,key in ((optimizer,'optimizer'),(scheduler,'scheduler'),(scaler,'scaler')):
        if obj is not None and key in ck:obj.load_state_dict(ck[key])
    return ck
