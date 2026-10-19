"""EEG spectral comparator with strictly past, label-free background reference.

Own experimental design. Baseline refreshes at the beginning of each 8-score
block, using up to 180 seconds ending BEFORE that block's first 4-second window.
No seizure labels, later samples, patient IDs or test statistics enter the input.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from modelo_memoria import TrainingSequences
from modelo_espectro import balanced_bce


@dataclass
class EEGBatch:
    windows: torch.Tensor
    history: torch.Tensor
    history_seconds: torch.Tensor

    def to(self, device):
        return EEGBatch(self.windows.to(device), self.history.to(device), self.history_seconds.to(device))


def log_maps(x, window):
    """4-second acquired windows -> 40 frequency bins x 13 frames."""
    shape = x.shape[:-1]
    transform = torch.stft(x.reshape(-1, 1024), 256, hop_length=64, win_length=256,
                           window=window, center=False, normalized=True, return_complex=True)
    return torch.log1p(transform.real.square()+transform.imag.square())[:, 1:41].reshape(*shape, 40, 13)


def append_history(array, first_end, mean, std, seconds=180):
    """Full chronological cache; left padding is masked, never mistaken for data."""
    stop = max(0, int(first_end)-4)
    start = max(0, stop-seconds)
    count = stop-start
    history = np.zeros((18, seconds*256), dtype=np.float32)
    if count:
        values = np.asarray(array[start*256:stop*256], dtype=np.float32).T
        history[:, -count*256:] = (values-mean[:, None])/std[:, None]
    return history, count


class FrequencyPreservingPool(nn.Module):
    """Keep the five frequency positions after the learned CNN; pool time only."""
    def forward(self, x):
        if x.shape[-2] != 5:
            raise ValueError('Expected five learned frequency positions.')
        n = x.shape[-1]
        return torch.stack([x[..., i*n//3:((i+1)*n+2)//3].mean(-1) for i in range(3)], -1)


class RelativeSpectrogramLSTM(nn.Module):
    def __init__(self, dropout, prior, config):
        super().__init__()
        self.history_length = int(config['relative']['history_seconds'])
        self.prior_weight = float(config['relative']['prior_seconds'])
        self.minimum_std = float(config['relative']['minimum_log_std'])
        self.soft_limit = float(config['relative']['relative_soft_limit'])
        self.top_k = int(config['relative']['top_k'])
        self.register_buffer('stft_window', torch.hann_window(256))
        mean = torch.tensor(prior['mean'], dtype=torch.float32)
        variance = torch.tensor(prior['variance'], dtype=torch.float32)
        if mean.shape != (40,) or variance.shape != (40,) or not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
            raise ValueError('Invalid TRAIN spectral prior.')
        self.register_buffer('prior_mean', mean)
        self.register_buffer('prior_variance', variance.clamp_min(0))
        self.channel_cnn = nn.Sequential(
            nn.Conv2d(2, 8, 3, stride=(2, 1), padding=1), nn.GroupNorm(4, 8), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=(2, 1), padding=1), nn.GroupNorm(4, 16), nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=(2, 1), padding=1), nn.GroupNorm(4, 24), nn.ReLU(),
            FrequencyPreservingPool())
        self.channel_projection = nn.Sequential(nn.Linear(24*5*3, 32), nn.ReLU())
        self.attention = nn.Sequential(nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1))
        self.fusion = nn.Sequential(nn.Linear(96, 128), nn.ReLU(), nn.Dropout(dropout))
        self.memory = nn.LSTM(128, 64, batch_first=True, bidirectional=False)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(64, 1))

    def time_frequency(self, x):
        return log_maps(x, self.stft_window)

    @torch.no_grad()
    def background(self, history, counts):
        if history.ndim != 3 or history.shape[1:] != (18, self.history_length*256):
            raise ValueError('Invalid past EEG buffer shape.')
        if torch.any((counts < 0) | (counts > self.history_length)):
            raise ValueError('Invalid history duration.')
        b = len(history)
        transform = torch.stft(history.reshape(b*18, -1), 256, hop_length=256, win_length=256,
                               window=self.stft_window, center=False, normalized=True, return_complex=True)
        values = torch.log1p(transform.real.square()+transform.imag.square())[:, 1:41]
        values = values.reshape(b, 18, 40, self.history_length)
        mask = (torch.arange(self.history_length, device=history.device)[None, :] >=
                self.history_length-counts[:, None]).float()[:, None, None, :]
        denominator = counts.float()[:, None, None]+self.prior_weight
        prior_mean = self.prior_mean[None, None, :]
        second_prior = self.prior_variance[None, None, :]+prior_mean.square()
        mean = ((values*mask).sum(-1)+self.prior_weight*prior_mean)/denominator
        second = ((values.square()*mask).sum(-1)+self.prior_weight*second_prior)/denominator
        std = (second-mean.square()).clamp_min(self.minimum_std**2).sqrt()
        return mean, std

    def paired_maps(self, batch):
        b, t = batch.windows.shape[:2]
        if batch.windows.shape[-2:] != (18, 1024):
            raise ValueError('Expected acquired four-second windows.')
        absolute = self.time_frequency(batch.windows.reshape(b*t, 18, 1024)).reshape(b, t, 18, 40, 13)
        mean, std = self.background(batch.history, batch.history_seconds)
        relative = (absolute-mean[:, None, :, :, None])/std[:, None, :, :, None]
        # Smooth bounding of the RELATIVE FEATURE only. No raw EEG or hours removed.
        relative = self.soft_limit*torch.tanh(relative/self.soft_limit)
        return absolute, relative

    def forward(self, batch, state=None, return_state=False):
        absolute, relative = self.paired_maps(batch)
        b, t = absolute.shape[:2]
        maps = torch.stack((absolute, relative), dim=3).reshape(b*t*18, 2, 40, 13)
        features = self.channel_projection(self.channel_cnn(maps).flatten(1)).reshape(b*t, 18, 32)
        mean = features.mean(1)
        top = features.topk(self.top_k, dim=1).values.mean(1)
        weights = self.attention(features).softmax(1)
        attended = (weights*features).sum(1)
        encoded = self.fusion(torch.cat((mean, top, attended), -1)).reshape(b, t, 128)
        sequence, next_state = self.memory(encoded, state)
        logits = self.classifier(sequence).squeeze(-1)
        return (logits, next_state) if return_state else logits


class RelativeSequences(TrainingSequences):
    def __init__(self, entries, lookup, quality, config):
        for entry in entries:
            if entry['role'] != 'train':
                continue
            cursor = 0
            for block in entry['segments']:
                if block['start'] != cursor or block['offset_samples'] != cursor*256:
                    raise ValueError('History requires a full chronological TRAIN cache.')
                cursor = block['stop']
            if cursor != int(lookup[entry['path']]['duration']):
                raise ValueError('Incomplete TRAIN cache.')
        super().__init__(entries, lookup, quality, config)

    def batch(self, indices, cache, stats, variant):
        if variant != 'espectro':
            raise ValueError('Unknown variant.')
        x, labels = super().batch(indices, cache, stats, 'memoria')
        mean = np.asarray(stats['mean'], dtype=np.float32)
        std = np.asarray(stats['std'], dtype=np.float32)+self.config['normalization_epsilon_uv']
        history, counts = [], []
        for rid, first, _, _, _ in self.samples[indices]:
            record = self.lookup[self.entries[int(rid)]['path']]
            h, count = append_history(cache.get(record), first, mean, std,
                                      self.config['relative']['history_seconds'])
            history.append(h); counts.append(count)
        return EEGBatch(x, torch.from_numpy(np.stack(history)), torch.tensor(counts)), labels


@torch.inference_mode()
def infer_record(model, array, ends, stats, config, device):
    if len(ends) and np.any(np.diff(ends) != config['advance_seconds']):
        raise ValueError('Reset at every gap.')
    steps, group_size = config['sequence_steps'], config['inference_batch_size']
    starts = list(range(0, len(ends), steps))
    probabilities = np.empty(len(ends), dtype=np.float32)
    mean = np.asarray(stats['mean'], dtype=np.float32)
    std = np.asarray(stats['std'], dtype=np.float32)+config['normalization_epsilon_uv']
    for offset in range(0, len(starts), group_size):
        group = starts[offset:offset+group_size]
        x = np.empty((len(group), steps, 18, 1024), dtype=np.float32)
        histories, counts, lengths = [], [], []
        for j, start in enumerate(group):
            actual = ends[start:start+steps]; length = len(actual)
            positions = actual[:, None]*256-1024+np.arange(1024)[None, :]
            x[j, :length] = np.asarray(array[positions]).transpose(0, 2, 1)
            x[j, length:] = x[j, length-1]
            h, n = append_history(array, int(actual[0]), mean, std, config['relative']['history_seconds'])
            histories.append(h); counts.append(n); lengths.append(length)
        x -= mean[None, None, :, None]; x /= std[None, None, :, None]
        batch = EEGBatch(torch.from_numpy(x), torch.from_numpy(np.stack(histories)), torch.tensor(counts)).to(device)
        scores = model(batch, state=None).sigmoid().cpu().numpy()
        for j, (start, length) in enumerate(zip(group, lengths)):
            probabilities[start:start+length] = scores[j, :length]
    if not np.isfinite(probabilities).all():
        raise ValueError('Non-finite predictions.')
    return probabilities
