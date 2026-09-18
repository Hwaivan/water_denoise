import torch
from ..diffusion.process import q_sample,combined_noise_target
from ..diffusion.losses import masked_mse
from ..utils.checkpoint import save_checkpoint,load_checkpoint
class Trainer:
    def __init__(self,model,schedule,optimizer,scheduler,ema,config,device):self.model,self.schedule,self.optimizer,self.scheduler,self.ema,self.c,self.device=model,schedule,optimizer,scheduler,ema,config,device;self.step=0;self.start=0;self.best=float('inf');self.scaler=torch.cuda.amp.GradScaler(enabled=config['training'].get('amp',True) and device.type=='cuda')
    def resume(self,path):
        ck=load_checkpoint(path,self.model,self.device,self.ema,self.optimizer,self.scheduler,self.scaler);self.start=ck['epoch']+1;self.step=ck['global_step'];self.best=ck['best_metric']
    def epoch(self,loader,train=True,max_batches=None):
        self.model.train(train);total=0.;count=0
        for batch in loader:
            clean=batch['clean'].to(self.device);noisy=batch['noisy'].to(self.device);lengths=batch['lengths'].to(self.device);t=torch.randint(1,self.schedule.num_steps+1,(clean.shape[0],),device=self.device);xt,eps,_=q_sample(self.schedule,clean,noisy,t);target=combined_noise_target(self.schedule,clean,noisy,eps,t)
            with torch.set_grad_enabled(train),torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):loss=masked_mse(self.model(xt,noisy,t),target,lengths)
            if train:
                self.optimizer.zero_grad(set_to_none=True);self.scaler.scale(loss).backward();self.scaler.unscale_(self.optimizer);torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.c['training'].get('grad_clip_norm',5.));self.scaler.step(self.optimizer);self.scaler.update();self.ema.update(self.model);self.step+=1
            total+=float(loss);count+=1
            if max_batches and count>=max_batches:break
        return total/max(1,count)
    def fit(self,train,valid):
        out=self.c['project']['output_dir']
        for e in range(self.start,self.c['training']['epochs']):
            tr=self.epoch(train,True);va=self.epoch(valid,False);self.scheduler.step(va);self.best=min(self.best,va);save_checkpoint(f'{out}/checkpoints/last.pt',self.model,self.ema,self.optimizer,self.scheduler,self.scaler,e,self.step,self.best,self.c)
            if va<=self.best:save_checkpoint(f'{out}/checkpoints/best.pt',self.model,self.ema,self.optimizer,self.scheduler,self.scaler,e,self.step,self.best,self.c)
            print(f'epoch={e+1} train_loss={tr:.6f} valid_loss={va:.6f}')
