import argparse,torch
from cdiffuse.utils.config import load_config
from cdiffuse.utils.checkpoint import load_checkpoint
from cdiffuse.data import load_audio,save_audio
from cdiffuse.factory import build_components
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');a=p.parse_args();c=load_config(a.config);d=torch.device(a.device);m,_,s=build_components(c);m.to(d);ck=load_checkpoint(a.checkpoint,m,d);m.load_state_dict(ck.get('ema_model',ck['model']));x=load_audio(a.input,c['data']['sample_rate']).to(d)[None];raw=s.sample(m,x).waveform;out=(1-c['evaluation']['noisy_mix_ratio'])*raw+c['evaluation']['noisy_mix_ratio']*x;save_audio(a.output,out,c['data']['sample_rate'])
if __name__=='__main__':main()
