"""Mejoras evaluables sin cambiar etiquetas, particiones ni EEG original."""
import numpy as np
import torch
from torch import nn
from modelo_cnn_bilstm import CNNBiLSTMSupCon


def causal_average(values, width):
    values=np.asarray(values,dtype=np.float64)
    if width<1:raise ValueError('Ancho positivo')
    sums=np.concatenate(([0.],np.cumsum(values,dtype=np.float64)))
    ends=np.arange(1,len(values)+1);starts=np.maximum(0,ends-width)
    return (sums[ends]-sums[starts])/(ends-starts)


def run_triggers(mask, steps):
    mask=np.asarray(mask,dtype=bool)
    bounds=np.diff(np.concatenate(([False],mask,[False])).astype(np.int8))
    starts=np.flatnonzero(bounds==1);stops=np.flatnonzero(bounds==-1)
    return starts[stops-starts>=steps]+steps-1


def causal_alarms(ends, scores, threshold, rule):
    scores=causal_average(scores,rule['smooth_seconds'])
    high=run_triggers(scores>=threshold,rule['open_steps'])
    low=run_triggers(scores<threshold*rule['close_ratio'],rule['close_steps'])
    actions=sorted([(int(i),1) for i in high]+[(int(i),0) for i in low])
    intervals=[];active=None
    for i,opened in actions:
        if opened and active is None:active=float(ends[i])
        elif not opened and active is not None:
            intervals.append([active,float(ends[i])]);active=None
    if active is not None:intervals.append([active,float(ends[-1])])
    return intervals


def event_row(record,prediction,threshold,rule):
    ends=prediction['ends'];first=float(ends[0]-1);last=float(ends[-1])
    events=[(a,b) for a,b in record['events'] if a>=first and b<=last]
    emitted=causal_alarms(ends,prediction['probabilities'],threshold,rule)
    remaining=set(range(len(events)));matches=[];delays=[]
    for i,(a,b) in enumerate(emitted):
        candidates=[j for j in sorted(remaining) if events[j][0]<=a<events[j][1]]
        if candidates:
            j=candidates[0];remaining.remove(j);matches.append([i,j]);delays.append(a-events[j][0])
    ictal=sum(max(0,min(b,last)-max(a,first)) for a,b in record['events'])
    return dict(path=record['path'],person=record['person'],threshold=float(threshold),rule=rule,
        events=len(events),partial_events=len(record['events'])-len(events),detected=len(matches),
        false_alarms=len(emitted)-len(matches),alarms=len(emitted),nonictal_seconds=last-first-ictal,
        evaluated_seconds=last-first,recorded_seconds=record['duration'],alarm_seconds=sum(b-a for a,b in emitted),
        delays=delays,alarm_intervals=emitted,matches=matches,coverage_interval=[first,last])


def choose_operating_point(points, budget, target):
    # El límite se aplica en validación, nunca se fuerza posteriormente en prueba.
    feasible=[p for p in points if p['total']['false_alarms_per_hour']<=budget]
    reaches=[p for p in feasible if p['total']['sensitivity']>=target]
    if reaches:return min(reaches,key=lambda p:(p['total']['false_alarms_per_hour'],p['total']['delay_median'] or 0,-p['threshold']))
    if feasible:return max(feasible,key=lambda p:(p['total']['sensitivity'],-p['total']['false_alarms_per_hour'],p['threshold']))
    raise ValueError('Incluir candidato sin alarmas para permitir límite verificable')


def shape_normalize(x):
    """Solo las muestras de la ventana ya adquirida. Sin etiquetas ni pacientes."""
    centered=x-x.mean(dim=-1,keepdim=True)
    scale=centered.square().mean(dim=-1,keepdim=True).sqrt().clamp_min(1e-5)
    return centered/scale


def choose_sensitivity_point(points,budget,max_alarm_fraction):
    feasible=[p for p in points if p['total']['false_alarms_per_hour']<=budget
              and p['total']['time_under_alarm']<=max_alarm_fraction]
    if not feasible:raise ValueError('Sin punto factible en validación')
    return max(feasible,key=lambda p:(p['total']['sensitivity'],-p['total']['false_alarms_per_hour'],
                                    -(p['total']['delay_median'] or 0),p['threshold']))


class RobustCNNBiLSTM(nn.Module):
    """CNN-BiLSTM-SupCon con forma normalizada y regularización."""
    def __init__(self,normalize=True,dropout=.2):
        super().__init__();self.normalize=normalize
        self.base=CNNBiLSTMSupCon()
        self.feature_norm=nn.LayerNorm(64)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,contrastive=False):
        if self.normalize:x=shape_normalize(x)
        features=self.feature_norm(self.base.cnn(x).transpose(1,2))
        sequence,_=self.base.bilstm(features)
        # La etiqueta corresponde al último segundo, no al promedio de los cuatro.
        h=self.dropout(self.base.representation(sequence[:,-16:].mean(dim=1)))
        logits=self.base.classifier(h).squeeze(-1)
        if contrastive:return logits,torch.nn.functional.normalize(self.base.projection(h),p=2,dim=1)
        return logits


class SpectralCNNBiLSTM(nn.Module):
    """Forma de onda y energía por bandas de la misma ventana ya adquirida.

    Las bandas son características, no un limpiador de artefactos. La salida
    se emite al terminar los cuatro segundos, igual que los otros candidatos.
    """
    def __init__(self,dropout=.2):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv1d(18,32,7,stride=4,padding=3),nn.ReLU(),
                               nn.Conv1d(32,64,5,stride=4,padding=2),nn.ReLU())
        self.wave_norm=nn.LayerNorm(64)
        self.spectral_projection=nn.Sequential(nn.Linear(90,64),nn.ReLU(),nn.LayerNorm(64))
        self.bilstm=nn.LSTM(128,64,batch_first=True,bidirectional=True)
        self.representation=nn.Sequential(nn.Linear(128,128),nn.ReLU(),nn.Dropout(dropout))
        self.classifier=nn.Linear(128,1)
        self.projection=nn.Linear(128,64)
        self.register_buffer('stft_window',torch.hann_window(256))
        frequency=torch.arange(129,dtype=torch.float32)
        masks=torch.stack([(frequency>=lo)&(frequency<hi) for lo,hi in
                           ((1,4),(4,8),(8,13),(13,41))]).float()
        self.register_buffer('band_masks',masks)

    def spectral_features(self,x):
        batch,channels,length=x.shape
        transform=torch.stft(x.reshape(batch*channels,length),n_fft=256,hop_length=64,
                             win_length=256,window=self.stft_window,center=False,
                             normalized=True,return_complex=True)
        power=transform.real.square()+transform.imag.square()
        band_power=torch.einsum('kf,bft->bkt',self.band_masks,power)
        total=band_power.sum(dim=1,keepdim=True).clamp_min(1e-6)
        relative=band_power/total
        features=torch.cat((relative,total.log()),dim=1)
        # 13 cuadros de un segundo, desplazados 0.25 s, sin relleno temporal.
        return features.reshape(batch,channels*5,-1).transpose(1,2)

    def forward(self,x,contrastive=False):
        spectrum=self.spectral_projection(self.spectral_features(x))
        wave=self.cnn(shape_normalize(x))
        wave=torch.nn.functional.adaptive_avg_pool1d(wave,spectrum.shape[1]).transpose(1,2)
        wave=self.wave_norm(wave)
        sequence,_=self.bilstm(torch.cat((wave,spectrum),dim=-1))
        h=self.representation(sequence[:,-4:].mean(dim=1))
        logits=self.classifier(h).squeeze(-1)
        if contrastive:return logits,torch.nn.functional.normalize(self.projection(h),p=2,dim=1)
        return logits


def make_model(variant,dropout=.2):
    if variant in ('shape','global'):
        return RobustCNNBiLSTM(normalize=variant=='shape',dropout=dropout)
    if variant=='spectral':return SpectralCNNBiLSTM(dropout=dropout)
    raise ValueError(f'Variante desconocida: {variant}')
