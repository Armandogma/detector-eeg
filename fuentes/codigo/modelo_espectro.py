"""Comparador propio CNN2D-LSTM sobre mapas STFT numéricos. Sin SupCon.

Cada mapa se calcula solo de la ventana adquirida. Los filtros convolucionales
se comparten entre derivaciones, que se concatenan en su orden antes de la LSTM.
"""
import torch
from torch import nn
from torch.nn import functional as F
from modelo_memoria import TrainingSequences


class FixedMapPool(nn.Module):
    """Promedios explícitos para evitar backward CUDA no determinista del pool adaptativo."""
    def forward(self,x):
        values=x.mean(dim=-2,keepdim=True);length=values.shape[-1]
        return torch.stack([values[...,i*length//3:((i+1)*length+2)//3].mean(-1) for i in range(3)],-1)


class SpectrogramLSTM(nn.Module):
    def __init__(self,dropout=.2):
        super().__init__()
        self.register_buffer('stft_window',torch.hann_window(256))
        self.channel_cnn=nn.Sequential(
            nn.Conv2d(1,8,3,stride=(2,1),padding=1),nn.GroupNorm(4,8),nn.ReLU(),
            nn.Conv2d(8,16,3,stride=(2,1),padding=1),nn.GroupNorm(4,16),nn.ReLU(),
            nn.Conv2d(16,24,3,stride=(2,1),padding=1),nn.GroupNorm(4,24),nn.ReLU(),
            FixedMapPool())
        self.channel_projection=nn.Sequential(nn.Linear(72,32),nn.ReLU())
        self.fusion=nn.Sequential(nn.Linear(18*32,128),nn.ReLU(),nn.Dropout(dropout))
        self.memory=nn.LSTM(128,64,batch_first=True,bidirectional=False)
        self.classifier=nn.Sequential(nn.Dropout(dropout),nn.Linear(64,1))

    def time_frequency(self,x):
        if x.ndim!=3 or x.shape[1:]!=(18,1024):
            raise ValueError('Se esperaban ventanas N x 18 x 1024.')
        n=x.shape[0]
        transform=torch.stft(x.reshape(n*18,1024),n_fft=256,hop_length=64,win_length=256,
            window=self.stft_window,center=False,normalized=True,return_complex=True)
        power=transform.real.square()+transform.imag.square()
        # Frecuencias lineales de 1 a 40 Hz, 13 posiciones temporales por ventana.
        return torch.log1p(power[:,1:41,:]).reshape(n,18,40,13)

    def encode(self,x):
        maps=self.time_frequency(x)
        features=self.channel_cnn(maps.reshape(-1,1,40,13)).flatten(1)
        channels=self.channel_projection(features).reshape(x.shape[0],18*32)
        return self.fusion(channels)

    def forward(self,x,state=None,return_state=False):
        if x.ndim!=4 or x.shape[-2:]!=(18,1024):
            raise ValueError('Se esperaba B x T x 18 x 1024, sin ventanas futuras.')
        batch,steps=x.shape[:2]
        encoded=self.encode(x.reshape(batch*steps,18,1024)).reshape(batch,steps,128)
        sequence,next_state=self.memory(encoded,state)
        logits=self.classifier(sequence).squeeze(-1)
        return (logits,next_state) if return_state else logits


def balanced_bce(logits,labels):
    mask=labels>=0;y=labels[mask];scores=logits[mask]
    positive,negative=y==1,y==0
    if not positive.any() or not negative.any():
        raise ValueError('El lote debe contener pasos reales de ambas clases.')
    losses=F.binary_cross_entropy_with_logits(scores,y,reduction='none')
    return .5*(losses[positive].mean()+losses[negative].mean())


class SpectralSequences(TrainingSequences):
    def batch(self,indices,cache,stats,variant):
        if variant!='espectro':raise ValueError('Variante desconocida.')
        return super().batch(indices,cache,stats,'memoria')
