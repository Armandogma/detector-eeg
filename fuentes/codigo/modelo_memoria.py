"""Ventanas adquiridas de 4 s y memoria causal entre ventanas consecutivas.

Implementación propia. Los estados pertenecen a cada secuencia, nunca a una persona
distinta o a una ventana posterior. No se importan pesos históricos.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from modelo_atencion import ChannelAttentionCNNBiLSTM
from modelo_cnn_bilstm import supervised_contrastive


def fixed_pool(values, frames):
    length = values.shape[-1]
    return torch.stack([
        values[..., i*length//frames:((i+1)*length+frames-1)//frames].mean(-1)
        for i in range(frames)], -1)


class MemoryModel(ChannelAttentionCNNBiLSTM):
    def __init__(self, dropout=.2):
        super().__init__(dropout)
        self.memory = nn.LSTM(128, 64, batch_first=True, bidirectional=False)
        self.memory_projection = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Dropout(dropout))

    def encode(self, x):
        wave, spectrum, _ = self.attended_features(x)
        spectrum = self.spectral_projection(spectrum)
        wave = fixed_pool(self.cnn(wave), spectrum.shape[1]).transpose(1, 2)
        wave = self.wave_norm(wave)
        within_window, _ = self.bilstm(torch.cat((wave, spectrum), dim=-1))
        return self.representation(within_window[:, -4:].mean(dim=1))

    def forward(self, x, contrastive=False, state=None, return_state=False):
        if x.ndim != 4 or x.shape[-2:] != (18, 1024):
            raise ValueError('Se esperaba B×T×18×1024, ventanas ya adquiridas.')
        batch, steps = x.shape[:2]
        embedding = self.encode(x.reshape(batch*steps, 18, 1024)).reshape(batch, steps, 128)
        sequence, next_state = self.memory(embedding, state)
        h = self.memory_projection(sequence)
        logits = self.classifier(h).squeeze(-1)
        output = (logits, F.normalize(self.projection(h), p=2, dim=-1)) if contrastive else logits
        return (output, next_state) if return_state else output


def labels_at(record, ends):
    labels = np.zeros(np.shape(ends), dtype=np.float32)
    for a, b in record['events']:
        labels[np.maximum(0, np.minimum(ends, b)-np.maximum(ends-1, a)) >= .5] = 1
    return labels


def sequence_losses(logits, z, labels, temperature, contrastive=True):
    mask = labels >= 0
    y = labels[mask]
    scores = logits[mask]
    positive, negative = y == 1, y == 0
    if not positive.any() or not negative.any():
        raise ValueError('El lote debe contener ambas clases reales, sin contar relleno.')
    losses = F.binary_cross_entropy_with_logits(scores, y, reduction='none')
    bce = .5*(losses[positive].mean()+losses[negative].mean())
    if contrastive:
        contrast = supervised_contrastive(z[mask], y, temperature)
    else:
        contrast = scores.sum()*0
    return bce, contrast


class TrainingSequences:
    def __init__(self, entries, lookup, quality, config):
        self.entries = [e for e in entries if e['role'] == 'train']
        self.lookup, self.config = lookup, config
        rows, people, near, event_samples = [], [], [], {}
        steps = config['sequence_steps']
        for rid, entry in enumerate(self.entries):
            record = lookup[entry['path']]
            for block in entry['segments']:
                ends = np.arange(block['start']+4, block['stop']+1, dtype=np.int32)
                if not len(ends):
                    continue
                lengths = np.minimum(steps, block['stop']-ends+1)
                last = ends+lengths-1
                npos = np.zeros(len(ends), dtype=np.int32)
                event_masks = []
                # Una etiqueta positiva requiere al menos 0.5 s ictal del segundo actual.
                for event, (a, b) in enumerate(record['events']):
                    ictal = np.maximum(0, np.minimum(ends, b)-np.maximum(ends-1, a)) >= .5
                    cumulative = np.r_[0, np.cumsum(ictal)]
                    count = cumulative[np.arange(len(ends))+lengths]-cumulative[np.arange(len(ends))]
                    npos += count
                    event_masks.append((event, count > 0))
                overlap = np.zeros(len(ends), dtype=bool)
                for a, b in record['events']:
                    overlap |= (last > a) & (ends-4 < b)
                allowed = (npos > 0) | ~overlap
                for a, b in quality.get(record['path'], []):
                    allowed &= ~((last > a) & (ends-4 < b))
                positions = np.flatnonzero(allowed)
                base = len(rows)
                for i in positions:
                    local = block.get('offset_samples', 0)+(int(ends[i])-block['start'])*256
                    rows.append([rid, int(ends[i]), local, int(npos[i] > 0), int(lengths[i])])
                    people.append(record['person'])
                    near.append(block['kind'] == 'episode')
                for event, mask in event_masks:
                    indices = base+np.flatnonzero(mask[positions])
                    if len(indices):
                        event_samples.setdefault((record['path'], event), []).extend(indices.tolist())
        self.samples = np.asarray(rows, dtype=np.int32).reshape(-1, 5)
        self.people = np.asarray(people)
        self.near = np.asarray(near, dtype=bool)
        self.person_names = sorted(set(people))
        expected = {(e['path'], i) for e in self.entries for i in range(len(lookup[e['path']]['events']))}
        if set(event_samples) != expected:
            raise ValueError('Crisis sin secuencia de entrenamiento. No se omitirán eventos.')
        self.pools = {}
        for person in self.person_names:
            neg = np.flatnonzero((self.people == person) & (self.samples[:, 3] == 0))
            events = [np.asarray(v, dtype=np.int64) for (path, _), v in sorted(event_samples.items())
                      if lookup[path]['person'] == person]
            if not len(neg) or not events:
                raise ValueError('Persona sin ambas clases de secuencia: '+person)
            self.pools[person] = dict(pos=events, neg=neg, near=neg[self.near[neg]], far=neg[~self.near[neg]])
        self.event_samples = event_samples

    def batches(self, epoch):
        rng = np.random.default_rng(self.config['seed']+epoch)
        size = self.config['training']['batch_size']
        if size % 2 or size < 4:
            raise ValueError('El lote debe ser par y contener al menos cuatro secuencias.')
        for _ in range(self.config['training']['epoch_sequences']//size):
            chosen = []
            for label in (0, 1):
                for _ in range(size//2):
                    pool = self.pools[str(rng.choice(self.person_names))]
                    if label:
                        options = pool['pos'][int(rng.integers(len(pool['pos'])))]
                    else:
                        kind = 'near' if rng.random() < self.config['training_negative_near_fraction'] else 'far'
                        options = pool[kind] if len(pool[kind]) else pool['neg']
                    chosen.append(int(rng.choice(options)))
            rng.shuffle(chosen)
            yield np.asarray(chosen, dtype=np.int64)

    def batch(self, indices, cache, stats, variant):
        if variant != 'memoria':
            raise ValueError('Este ensayo contiene una sola variante sin wavelet.')
        selected = self.samples[indices]
        steps = self.config['sequence_steps']
        x = np.empty((len(indices), steps, 18, 1024), dtype=np.float32)
        labels = np.full((len(indices), steps), -1, dtype=np.float32)
        for j, (rid, first, offset, _, length) in enumerate(selected):
            record = self.lookup[self.entries[int(rid)]['path']]
            array = cache.get(record)
            positions = int(offset)+np.arange(int(length))[:, None]*256-1024+np.arange(1024)[None, :]
            x[j, :length] = np.asarray(array[positions]).transpose(0, 2, 1)
            x[j, length:] = x[j, length-1]
            labels[j, :length] = labels_at(record, int(first)+np.arange(int(length)))
        x -= np.asarray(stats['mean'], dtype=np.float32)[None, None, :, None]
        x /= (np.asarray(stats['std'], dtype=np.float32)+self.config['normalization_epsilon_uv'])[None, None, :, None]
        return torch.from_numpy(x), torch.from_numpy(labels)


@torch.inference_mode()
def infer_record(model, array, ends, stats, config, device):
    """Bloques independientes de hasta 30 pasos. El relleno solo está al final."""
    if len(ends) and np.any(np.diff(ends) != config['advance_seconds']):
        raise ValueError('Hueco en la secuencia: reiniciar inferencia por intervalo continuo.')
    steps = config['sequence_steps']
    count = config['inference_batch_size']
    blocks = list(range(0, len(ends), steps))
    probabilities = np.empty(len(ends), dtype=np.float32)
    for group in range(0, len(blocks), count):
        starts = blocks[group:group+count]
        x = np.empty((len(starts), steps, 18, 1024), dtype=np.float32)
        lengths = []
        for j, start in enumerate(starts):
            actual = ends[start:start+steps]
            length = len(actual)
            positions = actual[:, None]*256-1024+np.arange(1024)[None, :]
            x[j, :length] = np.asarray(array[positions]).transpose(0, 2, 1)
            x[j, length:] = x[j, length-1]
            lengths.append(length)
        x -= np.asarray(stats['mean'], dtype=np.float32)[None, None, :, None]
        x /= (np.asarray(stats['std'], dtype=np.float32)+config['normalization_epsilon_uv'])[None, None, :, None]
        scores = model(torch.from_numpy(x).to(device), state=None).sigmoid().cpu().numpy()
        for j, (start, length) in enumerate(zip(starts, lengths)):
            probabilities[start:start+length] = scores[j, :length]
    if not np.isfinite(probabilities).all():
        raise ValueError('Probabilidades no finitas.')
    return probabilities
