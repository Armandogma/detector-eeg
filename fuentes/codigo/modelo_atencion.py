"""Atencion de 18 derivaciones para la CNN-BiLSTM-SupCon pediatrica.

Adaptacion pequena de recalibracion de canales. Usa solo la ventana ya
adquirida y no clasifica artefactos. Inicialmente equivale al modelo base.
"""
import torch
from torch import nn
from torch.nn import functional as F
from ajustes_modelo import SpectralCNNBiLSTM,shape_normalize


class ChannelAttentionCNNBiLSTM(SpectralCNNBiLSTM):
    def __init__(self,dropout=.2):
        super().__init__(dropout)
        # Mantener la secuencia aleatoria del modelo base para la comparacion
        # emparejada. Los pesos originales reciben la misma inicializacion.
        before=torch.get_rng_state()
        self.channel_gate=nn.Sequential(nn.Linear(18,6,bias=False),nn.SiLU(),nn.Linear(6,18,bias=False))
        nn.init.zeros_(self.channel_gate[-1].weight)
        torch.set_rng_state(before)

    def channel_weights(self,x):
        if x.ndim!=3 or x.shape[1:]!=(18,1024):
            raise ValueError('Se esperan 18 derivaciones y cuatro segundos a 256 Hz')
        rms=x.float().square().mean(-1).clamp_min(1e-12).sqrt()
        descriptor=torch.log1p(rms)
        return 2*torch.sigmoid(self.channel_gate(descriptor))

    def attended_features(self,x):
        weights=self.channel_weights(x)
        spectrum=self.spectral_features(x)
        batch,frames,_=spectrum.shape
        # Pesar las cinco caracteristicas de cada derivacion. Aplicar el peso
        # antes de calcular potencias relativas lo cancelaria en esas razones.
        spectrum=(spectrum.reshape(batch,frames,18,5)*weights[:,None,:,None]).reshape(batch,frames,90)
        # La forma se normaliza primero. Si se ponderara antes, la normalizacion
        # por canal cancelaria el peso aprendido de la rama de ondas.
        wave=shape_normalize(x)*weights[:,:,None]
        return wave,spectrum,weights

    def forward(self,x,contrastive=False):
        wave,spectrum,_=self.attended_features(x)
        spectrum=self.spectral_projection(spectrum)
        wave=self.cnn(wave)
        wave=F.adaptive_avg_pool1d(wave,spectrum.shape[1]).transpose(1,2)
        wave=self.wave_norm(wave)
        sequence,_=self.bilstm(torch.cat((wave,spectrum),dim=-1))
        h=self.representation(sequence[:,-4:].mean(dim=1))
        logits=self.classifier(h).squeeze(-1)
        if contrastive:return logits,F.normalize(self.projection(h),p=2,dim=1)
        return logits
