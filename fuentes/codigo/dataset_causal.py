"""Filtro continuo por bloques y ventanas en disco; no lee EEG futuro."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfilt

from rutas import REPO, DATASET

PREPARED = REPO / 'preparados'


def configuration():
    return json.loads((REPO / 'codigo/configuracion.json').read_text(encoding='utf-8'))


def load_manifest():
    return json.loads((REPO / 'analisis/manifiesto_dataset.json').read_text(encoding='utf-8'))


def identifier(record):
    return record['path'].replace('/', '__').replace('.edf', '')


def signature(record, config):
    info = {k: record[k] for k in ('path', 'header_sha256', 'file_bytes', 'channel_map')}
    info.update({k: config[k] for k in ('filter_design_order', 'filter_low_hz', 'filter_high_hz',
                                      'fs', 'filter_initial_state')})
    info['implementation_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()


def read_calibrated(record, start, stop):
    """Lee segundos enteros y resta señales CS2 después de calibrar."""
    h = record['header']
    data = np.memmap(DATASET / record['path'], mode='r', dtype='<i2', offset=h['hbytes'],
                     shape=(h['records'], h['n'], 256))
    needed = sorted({i for pair in record['channel_map'] for i in pair})
    raw = np.asarray(data[start:stop, needed, :], dtype=np.float64).transpose(1, 0, 2).reshape(len(needed), -1)
    denominator = np.array(h['dmax']) - np.array(h['dmin'])
    if any(denominator[i] == 0 for i in needed):
        raise ValueError('Canal seleccionado sin calibración válida')
    gain = np.divide(np.array(h['pmax']) - np.array(h['pmin']), denominator,
                     out=np.zeros(len(denominator)), where=denominator != 0)
    bias = np.array(h['pmin']) - gain * np.array(h['dmin'])
    if any(h['units'][i] != 'uV' for i in needed):
        raise ValueError('Se esperaba escala uV')
    raw = raw * gain[needed, None] + bias[needed, None]
    lookup = dict(zip(needed, raw))
    out = [lookup[pair[0]] if len(pair) == 1 else lookup[pair[0]] - lookup[pair[1]]
           for pair in record['channel_map']]
    return np.stack(out, axis=0)


def prepare(record, config):
    if not record['eligible']:
        raise ValueError(f"Registro no elegible: {record['path']}")
    path = DATASET / record['path']
    with path.open('rb') as f:
        actual_hash = hashlib.sha256(f.read(record['header']['hbytes'])).hexdigest()
    if path.stat().st_size != record['file_bytes'] or actual_hash != record['header_sha256']:
        raise ValueError('EDF cambió desde el manifiesto')
    PREPARED.mkdir(exist_ok=True)
    target = PREPARED / f'{identifier(record)}.npy'
    meta = target.with_suffix('.json')
    fingerprint = signature(record, config)
    if target.exists() and meta.exists():
        info = json.loads(meta.read_text())
        if info['signature'] == fingerprint:
            arr = np.load(target, mmap_mode='r')
            if arr.shape == (int(record['duration'] * 256), 18) and arr.dtype == np.float32:
                return target
    # Escribe temporal, publica solo al terminar. Un bloqueo no deja caché parcial válida.
    temp = target.with_suffix('.partial.npy')
    out = np.lib.format.open_memmap(temp, mode='w+', dtype='float32',
                                  shape=(int(record['duration'] * 256), 18))
    sos = butter(config['filter_design_order'], [config['filter_low_hz'], config['filter_high_hz']],
                 btype='bandpass', fs=256, output='sos')
    state = np.zeros((len(sos), 18, 2), dtype=np.float64)
    checksum = hashlib.sha256()
    for start in range(0, int(record['duration']), config['preprocessing_block_seconds']):
        stop = min(start + config['preprocessing_block_seconds'], int(record['duration']))
        x = read_calibrated(record, start, stop)
        filtered, state = sosfilt(sos, x, axis=1, zi=state)
        values = np.ascontiguousarray(filtered.T, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError('Filtro produjo valores no finitos')
        out[start * 256:stop * 256] = values
        checksum.update(values.tobytes())
    out.flush(); del out
    os.replace(temp, target)
    meta.write_text(json.dumps(dict(signature=fingerprint, source=record['path'],
                                   filtered_samples_sha256=checksum.hexdigest(),
                                   dtype='float32', units='uV', causal=True,
                                   initial_state='zero', channel_map=record['channel_map']), indent=2), encoding='utf-8')
    return target


def normalization(records, config):
    """Estadísticos exclusivamente de los EDF de entrenamiento, sin balanceo."""
    count = 0
    mean, m2 = np.zeros(18), np.zeros(18)
    for r in records:
        data = np.load(prepare(r, config), mmap_mode='r')
        for start in range(0, len(data), 60 * 256):
            values = np.array(data[start:start + 60 * 256], dtype=np.float64)
            n, block_mean = len(values), values.mean(axis=0)
            block_m2 = ((values - block_mean) ** 2).sum(axis=0)
            delta = block_mean - mean
            m2 += block_m2 + delta**2 * count * n / (count + n)
            mean += delta * n / (count + n)
            count += n
    return dict(mean=mean.tolist(), std=np.sqrt(m2 / count).tolist(), sample_count=count,
                fitted_records=[r['path'] for r in records], epsilon=config['normalization_epsilon_uv'])


def record_index(record, config):
    ends = np.arange(config['window_seconds'], int(record['duration']) + 1,
                     config['advance_seconds'], dtype=np.int32)
    labels = np.zeros(len(ends), dtype=np.int8)
    groups = np.full(len(ends), -1, dtype=np.int32)
    for event, (a, b) in enumerate(record['events']):
        overlap = np.maximum(0, np.minimum(ends, b) - np.maximum(ends - 1, a))
        mask = overlap >= .5
        labels[mask] = 1; groups[mask] = event
    return ends, labels, groups


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, records, stats, config):
        self.records, self.config = records, config
        self.mean = np.asarray(stats['mean'], dtype=np.float32)
        self.scale = np.asarray(stats['std'], dtype=np.float32) + stats['epsilon']
        self.data, self.ends, labels, groups, people = [], [], [], [], []
        people_ids = {p: i for i, p in enumerate(sorted({r['person'] for r in records}))}
        for i, r in enumerate(records):
            self.data.append(np.load(prepare(r, config), mmap_mode='r'))
            ends, lab, event = record_index(r, config)
            self.ends.append(ends)
            labels.append(lab)
            groups.append(np.where(lab == 1, i * 1000 + event, -(people_ids[r['person']] + 1)))
            people.append(np.full(len(ends), i, dtype=np.int32))
        self.labels = np.concatenate(labels)
        self.groups = np.concatenate(groups)
        self.record_ids = np.concatenate(people)
        self.offsets = np.cumsum([0] + [len(x) for x in self.ends])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        rid = int(self.record_ids[index])
        local = index - self.offsets[rid]
        end = int(self.ends[rid][local]) * 256
        values = np.array(self.data[rid][end - 1024:end], dtype=np.float32)
        values = ((values - self.mean) / self.scale).T.copy()
        return torch.from_numpy(values), torch.tensor(float(self.labels[index]))


class BalancedBatches(torch.utils.data.Sampler):
    """1:1. Distribuye positivos entre crisis y negativos entre personas."""
    def __init__(self, dataset, batch_size, epoch_windows, seed):
        if batch_size < 4 or batch_size % 2:
            raise ValueError('Lote par >=4')
        self.batch_size, self.steps, self.seed = batch_size, max(1, epoch_windows // batch_size), seed
        self.epoch = 0
        self.pools = []
        for label in (0, 1):
            groups = sorted(set(dataset.groups[dataset.labels == label].tolist()))
            if not groups:
                raise ValueError('Se necesitan ambas clases en entrenamiento')
            self.pools.append([np.flatnonzero((dataset.labels == label) & (dataset.groups == group))
                               for group in groups])

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch); self.epoch += 1
        for _ in range(self.steps):
            batch = []
            for pools in self.pools:
                selected = np.resize(rng.permutation(len(pools)), self.batch_size // 2)
                # Muestreo por evento/persona, no únicamente una crisis con ventanas vecinas.
                for group in selected:
                    batch.append(int(rng.choice(pools[group])))
            rng.shuffle(batch)
            yield batch
