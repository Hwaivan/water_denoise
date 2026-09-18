import argparse,random
import numpy as np
import torch
from cdiffuse.utils.config import load_config
from cdiffuse.data import CDiffuSEDataset,build_dataloader
from cdiffuse.factory import build_components
from cdiffuse.utils.ema import EMA
from cdiffuse.engine import Trainer
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume');p.add_argument('--device',choices=('cpu','cuda'),default='cuda' if torch.cuda.is_available() else 'cpu');a=p.parse_args();c=load_config(a.config);seed=c['training']['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);device=torch.device(a.device)
    model,schedule,_=build_components(c);model.to(device);ema=EMA(model,c['training']['ema_decay']);opt=torch.optim.Adam(model.parameters(),lr=c['training']['learning_rate']);sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt);trainer=Trainer(model,schedule,opt,sch,ema,c,device)
    if a.resume:trainer.resume(a.resume)
    trainer.fit(build_dataloader(CDiffuSEDataset(c['data'],'train',seed),c,True),build_dataloader(CDiffuSEDataset(c['data'],'valid',seed+1),c,False))
if __name__=='__main__':main()
