"""Hipótesis propia 2 s + 4 s, salida rápida y muestreo por evento solo TRAIN.

No altera los EDF ni las etiquetas. No atribuye un diagnóstico clínico a ningún
caso y no garantiza mejorar. La rama de 4 s conserva el extractor anterior.
"""
import heapq
import numpy as np
from scipy.signal import stft
import torch
from torch import nn
from torch.nn import functional as F
from modelo_adaptativo import extract_features_from_filtered as extract_four
from modelo_adaptativo import MagnitudePreservingSeizureNet, labels_at


def past_reference(values, width=180, gap=2, floor=.05):
    """La última ventana del fondo termina al comenzar la ventana actual.

    Cada columna resume 2 segundos. Si no hay pasado se usa media 0/std 1
    deterministas. No se completa el inicio con muestras futuras ni etiquetas.
    """
    a = np.asarray(values, dtype=np.float64)
    n = a.shape[-1]
    hi = np.maximum(0, np.arange(n)-gap+1)
    lo = np.maximum(0, hi-width)
    count = hi-lo
    sums = np.concatenate((np.zeros(a.shape[:-1]+(1,)), np.cumsum(a, axis=-1)), axis=-1)
    squares = np.concatenate((np.zeros(a.shape[:-1]+(1,)), np.cumsum(a*a, axis=-1)), axis=-1)
    mean = (sums[..., hi]-sums[..., lo])/np.maximum(count, 1)
    var = (squares[..., hi]-squares[..., lo])/np.maximum(count, 1)-mean*mean
    std = np.sqrt(np.maximum(var, floor*floor))
    std[..., count == 0] = 1.
    return ((a-mean)/std).astype(np.float32), mean.astype(np.float32)


def extract_two(filtered, config):
    fs = int(config['fs'])
    if fs != 256 or filtered.shape[0] != 18:
        raise ValueError('Se requieren 18 derivaciones a 256 Hz.')
    duration = filtered.shape[1]//fs
    x = 500*np.tanh(np.asarray(filtered[:, :duration*fs], dtype=np.float32)/500)
    seconds = x.reshape(18, duration, fs)
    rms = np.log1p(np.sqrt((seconds**2).mean(-1)))
    line = np.log1p(np.abs(np.diff(seconds, axis=-1)).mean(-1))
    mobility = np.sqrt((np.diff(seconds, axis=-1)**2).mean(-1)/((seconds**2).mean(-1)+1e-5))
    _, _, spectrum = stft(x, fs=fs, window='hann', nperseg=fs, noverlap=0,
                          boundary=None, padded=False, axis=-1)
    power = (np.abs(spectrum[:, 1:41])**2).astype(np.float32)
    log = np.log1p(power)
    density = power/(power.sum(1, keepdims=True)+1e-8)
    entropy = -(density*np.log(density+1e-8)).sum(1)/np.log(40)
    ratio = np.log1p(power[:, 7:].sum(1))-np.log1p(power[:, :7].sum(1))
    ratio2 = np.log1p(power[:, 12:].sum(1))-np.log1p(power[:, :7].sum(1))
    mean2 = .5*(log[..., 1:]+log[..., :-1])
    delta = log[..., 1:]-log[..., :-1]
    aux = np.stack((rms, line, mobility, entropy, ratio, ratio2), axis=1)
    aux2 = .5*(aux[..., 1:]+aux[..., :-1])
    options = dict(width=int(config['adaptive']['baseline_window_seconds']), gap=2,
                   floor=float(config['adaptive']['minimum_baseline_std']))
    z, background = past_reference(mean2, **options)
    aux_z, _ = past_reference(aux2, **options)
    z = np.clip(z, -6, 10); aux_z = np.clip(aux_z, -6, 10)
    relative = mean2-mean2.mean(1, keepdims=True)-background+background.mean(1, keepdims=True)
    features = []
    for a, b in ((0, 2), (2, 4), (4, 7), (7, 10), (10, 14), (14, 21), (21, 29), (29, 40)):
        features.extend((z[:, a:b].mean(1), np.clip(z[:, a:b], 0, 10).mean(1),
                         relative[:, a:b].mean(1), mean2[:, a:b].mean(1)/10,
                         delta[:, a:b].mean(1)))
    fast, slow = z[:, 7:].mean(1), z[:, :7].mean(1)
    features.extend((fast-slow, np.clip(fast, 0, 10)*np.clip(-slow, 0, 6)))
    features.extend(aux_z[:, k] for k in range(6))
    # Las columnas originales terminan en 2..duración. Alinear con 4..duración.
    return np.stack(features, axis=-1).transpose(1, 0, 2)[2:].astype(np.float32)


def extract_features(filtered, config):
    long, ends = extract_four(filtered, config)
    short = extract_two(filtered, config)
    if short.shape != long.shape or long.shape[-1] != 48:
        raise ValueError('Las dos escalas deben estar alineadas en los mismos segundos.')
    result = np.concatenate((long, short), axis=-1)
    if not np.isfinite(result).all():
        raise ValueError('Características no finitas.')
    return result, ends


class BriefSeizureNet(nn.Module):
    """TCN–GRU de 4 s más clasificador instantáneo sobre características de 2 s."""
    def __init__(self, dropout=.25, config=None):
        super().__init__()
        self.slow = MagnitudePreservingSeizureNet(dropout, {'adaptive': {'n_features': 48}})
        self.fast_scale = nn.Parameter(torch.full((48,), .25))
        self.fast_channels = nn.Sequential(nn.Linear(48, 48), nn.GELU(), nn.Dropout(dropout),
                                           nn.Linear(48, 24), nn.GELU())
        self.fast_attention = nn.Linear(24, 1)
        self.fast_head = nn.Sequential(nn.Linear(72, 48), nn.GELU(), nn.Dropout(dropout), nn.Linear(48, 1))
        self.mix = nn.Parameter(torch.tensor(0.))

    def forward(self, x, return_heads=False):
        if x.ndim != 4 or x.shape[-2:] != (18, 96):
            raise ValueError('Se esperaba (lote, pasos, 18, 96).')
        slow = self.slow(x[..., :48])
        h = self.fast_channels(x[..., 48:]*self.fast_scale)
        attended = (self.fast_attention(h).softmax(2)*h).sum(2)
        pooled = torch.cat((h.mean(2), h.topk(4, dim=2).values.mean(2), attended), dim=-1)
        fast = self.fast_head(pooled).squeeze(-1)
        alpha = self.mix.sigmoid()
        combined = (1-alpha)*slow+alpha*fast
        return (combined, slow, fast) if return_heads else combined


def loss_one(logits, labels, onset_weights):
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    positive = labels >= .5
    pos = ((loss*onset_weights)[positive].sum()/onset_weights[positive].sum().clamp_min(1)
           if positive.any() else logits.sum()*0)
    neg = ((loss*(1+4*logits.sigmoid().detach()**2))[~positive].mean()
           if (~positive).any() else logits.sum()*0)
    return .30*pos+.70*neg if positive.any() and (~positive).any() else pos+neg


def training_loss(heads, labels, onset_weights):
    combined, slow, fast = heads
    return .70*loss_one(combined, labels, onset_weights)+.15*loss_one(slow, labels, onset_weights)+.15*loss_one(fast, labels, onset_weights)


@torch.inference_mode()
def infer_record(model, array, config, device):
    """Lotes acotados, cobertura exacta y salida de prefijos invariante al futuro."""
    model.eval()
    length, stride = int(config['sequence_steps']), int(config['adaptive']['inference_stride'])
    keep, n = length-stride, len(array)
    if n == 0:
        return np.empty(0, dtype=np.float32)
    starts = [0]+list(range(stride, max(stride, n-keep), stride))
    output = np.empty(n, dtype=np.float32)
    written = np.zeros(n, dtype=np.uint8)
    batch = int(config['inference_batch_size'])
    for offset in range(0, len(starts), batch):
        group = starts[offset:offset+batch]
        x = np.zeros((len(group), length, 18, 96), dtype=np.float32)
        counts = []
        for j, start in enumerate(group):
            count = min(length, n-start)
            x[j, :count] = np.asarray(array[start:start+count], dtype=np.float32)
            counts.append(count)
        scores = model(torch.from_numpy(x).to(device)).sigmoid().cpu().numpy()
        for j, (start, count) in enumerate(zip(group, counts)):
            first = 0 if start == 0 else keep
            output[start+first:start+count] = scores[j, first:count]
            written[start+first:start+count] += 1
    if not np.all(written == 1) or not np.isfinite(output).all():
        raise ValueError('Cobertura o puntuaciones inválidas.')
    return output


class TrainingSampler:
    """Personas y eventos uniformes; ningún dato de validación o prueba entra aquí."""
    def __init__(self, entries, lookup, config):
        self.config, self.lookup = config, lookup
        self.entries = [e for e in entries if e['role'] == 'train']
        self.length = int(config['sequence_steps'])
        self.people = sorted({e['person'] for e in self.entries})
        self.events = {p: [] for p in self.people}
        self.negative = {p: [] for p in self.people}
        self.hard = {p: [] for p in self.people}
        for entry in self.entries:
            record = lookup[entry['path']]
            n = int(record['duration'])-3
            if n < self.length:
                raise ValueError('Registro demasiado corto para el protocolo.')
            for a, b in record['events']:
                self.events[record['person']].append((record['path'], a, b))
            labels = labels_at(record, np.arange(4, int(record['duration'])+1))
            sums = np.concatenate(([0], np.cumsum(labels, dtype=np.int32)))
            valid = np.flatnonzero(sums[self.length:]-sums[:-self.length] == 0)[::2].astype(np.int32)
            if len(valid):
                self.negative[record['person']].append((record['path'], valid))
        if any(not self.events[p] or not self.negative[p] for p in self.people):
            raise ValueError('Se necesitan crisis y fondo en cada persona de TRAIN.')
        self.negative_counts = {p: np.cumsum([len(v) for _, v in self.negative[p]]) for p in self.people}

    def _negative(self, person, rng):
        if self.hard[person] and rng.random() < .5:
            return self.hard[person][int(rng.integers(len(self.hard[person])))]
        cumulative = self.negative_counts[person]
        rank = int(rng.integers(cumulative[-1]))
        i = int(np.searchsorted(cumulative, rank, side='right'))
        previous = 0 if i == 0 else cumulative[i-1]
        path, starts = self.negative[person][i]
        return path, int(starts[rank-previous])

    def batch(self, rng, cache, augment=True):
        items = []
        # Cada lote incluye cada persona TRAIN una vez como ancla positiva.
        for person in rng.permutation(self.people):
            events = self.events[person]
            path, a, b = events[int(rng.integers(len(events)))]
            first, last = int(np.floor(a))+1, int(np.floor(b))
            if rng.random() < self.config['brief']['onset_sampling_fraction']:
                last = min(last, first+self.config['brief']['onset_seconds']-1)
            end = int(rng.integers(first, last+1))
            start = end-4-int(rng.integers(self.length))
            start = max(0, min(int(self.lookup[path]['duration'])-3-self.length, start))
            items.append((path, start))
            items.extend((self._negative(person, rng), self._negative(person, rng)))
        x, y, weights = [], [], []
        for path, start in items:
            record = self.lookup[path]
            x.append(np.array(cache.get(record)[start:start+self.length], dtype=np.float32))
            ends = np.arange(4+start, 4+start+self.length)
            labels = labels_at(record, ends)
            w = np.ones(self.length, dtype=np.float32)
            for a, b in record['events']:
                w[(ends > a) & (ends <= min(b, a+self.config['brief']['onset_seconds']))] = self.config['brief']['onset_loss_weight']
            y.append(labels); weights.append(w)
        array = np.stack(x)
        if augment:
            array *= rng.uniform(.85, 1.15, (len(items), 1, 1, 1)).astype(np.float32)
        return torch.from_numpy(array), torch.from_numpy(np.stack(y)), torch.from_numpy(np.stack(weights))

    def mine(self, model, cache, device):
        limit = int(self.config['brief']['hard_per_person'])
        heaps = {p: [] for p in self.people}
        for entry in self.entries:
            record = self.lookup[entry['path']]
            scores = infer_record(model, cache.get(record), self.config, device)
            labels = labels_at(record, np.arange(4, int(record['duration'])+1))
            candidates = np.flatnonzero((scores > self.config['adaptive']['hard_mining_threshold']) & (labels == 0))
            candidates = candidates[np.argsort(-scores[candidates], kind='stable')[:limit]]
            seen = set()
            for idx in candidates:
                start = max(0, min(len(labels)-self.length, int(idx)-self.length+1))
                if start in seen or labels[start:start+self.length].any():
                    continue
                seen.add(start)
                item = (float(scores[idx]), record['path'], start)
                heap = heaps[record['person']]
                if len(heap) < limit:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
        self.hard = {p: [(path, start) for _, path, start in sorted(values, reverse=True)] for p, values in heaps.items()}
        return sum(len(v) for v in self.hard.values())

    def state_dict(self):
        return {p: [[path, int(start)] for path, start in values] for p, values in self.hard.items()}

    def load_state_dict(self, value):
        if set(value) != set(self.people):
            raise ValueError('La minería guardada contiene otras personas.')
        train_paths = {e['path'] for e in self.entries}
        for person, values in value.items():
            for path, start in values:
                if path not in train_paths or self.lookup[path]['person'] != person or not isinstance(start, int):
                    raise ValueError('Muestra difícil ajena a TRAIN.')
                record = self.lookup[path]
                if start < 0 or start+self.length > int(record['duration'])-3:
                    raise ValueError('Muestra difícil fuera de rango.')
                if labels_at(record, np.arange(4+start, 4+start+self.length)).any():
                    raise ValueError('Muestra difícil con etiquetas positivas.')
        self.hard = {p: [(path, start) for path, start in values] for p, values in value.items()}
