import torch
from torch import nn
import torch.nn.functional as F

class NoisySpectrumConditioner(nn.Module):
    """Magnitude-STFT encoder: input [B,1,L], output [B,C,L]."""
    def __init__(self,channels=128,n_fft=1024,win_length=1024,hop_length=256):
        super().__init__(); self.n_fft,self.win_length,self.hop_length=n_fft,win_length,hop_length
        self.register_buffer("window",torch.hann_window(win_length),persistent=False)
        self.encoder=nn.Sequential(nn.Conv1d(n_fft//2+1,channels,3,padding=1),nn.SiLU(),nn.Conv1d(channels,channels,3,padding=1))
    def forward(self,noisy,length=None):
        if noisy.ndim==3: noisy=noisy[:,0]
        if noisy.ndim!=2: raise ValueError("condition waveform must be [B,L] or [B,1,L]")
        length=noisy.shape[-1] if length is None else int(length)
        window=self.window.to(device=noisy.device,dtype=noisy.dtype)
        spec=torch.stft(noisy,n_fft=self.n_fft,hop_length=self.hop_length,win_length=self.win_length,window=window,return_complex=True,onesided=True)
        encoded=self.encoder(spec.abs())
        return F.interpolate(encoded,size=length,mode="linear",align_corners=False)
