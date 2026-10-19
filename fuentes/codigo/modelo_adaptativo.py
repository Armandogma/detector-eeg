"""Arquitectura causal-adaptativa preservadora de magnitud (TCN-GRU + Atención Espacial).

Diseñada para resolver:
1) Crisis con atenuación de voltaje / desincronización de ondas lentas (p. ej. chb14 y chb06),
   mediante normalización espectral causal por bin de frecuencia (1-40 Hz) respecto a los
   180 segundos previos del propio registro (causal_zscore), cambio espectral relativo
   intraventana (dRel), contraste rápido-vs-lento (8-40 Hz vs 1-7 Hz) y descriptores
   morfológicos causales (longitud de línea, RMS, movilidad de Hjorth, entropía espectral).
2) Descalibración de umbral entre pacientes (p. ej. chb09 vs chb23), eliminando las capas
   LayerNorm por instante temporal que destruían la magnitud del puntaje Z causal y amplificaban
   fluctuaciones basales silenciosas.
3) Deriva de estado oculto y reinicios a mitad de crisis, combinando bloques TCN causales
   dilatados (d = 1, 2, 4, 8) con inferencia GRU causal en ventanas solapadas (L = 24 s,
   avance = 12 s).

No utiliza etiquetas futuras, estadísticos de prueba, identificadores de paciente ni SMOTE.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, stft
import torch
from torch import nn
from torch.nn import functional as F

import dataset_causal as causal


def design_filter(config: dict) -> np.ndarray:
    return butter(
        int(config["filter_design_order"]),
        [float(config["filter_low_hz"]), float(config["filter_high_hz"])],
        btype="bandpass",
        fs=int(config["fs"]),
        output="sos",
    )


def causal_zscore(
    arr: np.ndarray,
    window_sec: int = 180,
    gap_sec: int = 4,
    min_std: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalización Z estrictamente causal sobre los window_sec segundos previos [t - gap - W, t - gap)."""
    T = arr.shape[-1]
    arr64 = arr.astype(np.float64)
    csum = np.cumsum(arr64, axis=-1)
    csum2 = np.cumsum(arr64 ** 2, axis=-1)
    idx = np.arange(T)
    hi = np.maximum(1, idx - gap_sec)
    lo = np.maximum(0, hi - window_sec)
    cnt = np.maximum(1, hi - lo)
    shape_exp = (1,) * (arr.ndim - 1) + (T,)
    cnt = cnt.reshape(shape_exp)
    hi_s = csum[..., hi - 1]
    lo_s = np.where((lo > 0).reshape(shape_exp), csum[..., np.maximum(0, lo - 1)], 0.0)
    hi_s2 = csum2[..., hi - 1]
    lo_s2 = np.where((lo > 0).reshape(shape_exp), csum2[..., np.maximum(0, lo - 1)], 0.0)
    mean_b = (hi_s - lo_s) / cnt
    std_b = np.sqrt(np.maximum((hi_s2 - lo_s2) / cnt - mean_b ** 2, min_std ** 2))
    return ((arr64 - mean_b) / std_b).astype(np.float32), mean_b.astype(np.float32)


def labels_at(record: dict, ends: np.ndarray) -> np.ndarray:
    """Etiqueta positiva si >= 0.5 s del último segundo [end-1, end] solapa con una crisis anotada."""
    ends = np.asarray(ends, dtype=np.int32)
    labels = np.zeros(len(ends), dtype=np.float32)
    for a, b in record["events"]:
        overlap = np.minimum(ends, b) - np.maximum(ends - 1, a)
        labels[overlap >= 0.5] = 1.0
    return labels


def extract_features_from_filtered(
    filtered: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Extrae los 48 descriptores espectrales y morfológicos causales por canal y segundo.

    Entrada:
        filtered: arreglo float32 de forma (18, dur * 256) filtrado causalmente (0.5-40 Hz).
    Salida:
        X: arreglo float32 de forma (T, 18, 48) para cada ventana de 4 s terminada en ends.
        ends: arreglo int32 de forma (T,) con los segundos finales 4..dur.
    """
    fs = int(config["fs"])
    dur = filtered.shape[1] // fs
    if dur < int(config["window_seconds"]):
        raise ValueError("Registro más corto que la ventana de 4 segundos.")
    win_sec = int(config["adaptive"]["baseline_window_seconds"])
    gap_sec = int(config["adaptive"]["baseline_gap_seconds"])
    min_std = float(config["adaptive"]["minimum_baseline_std"])

    # Acotamiento suave fisiológico para evitar que artefactos de desconexión dominen la varianza
    x_clipped = 500.0 * np.tanh(filtered[:, : dur * fs].astype(np.float32) / 500.0)
    sec_reshaped = x_clipped.reshape(18, dur, fs)

    # 1. Descriptores morfológicos en dominio del tiempo por segundo (18, dur)
    rms_1s = np.log1p(np.sqrt(np.mean(sec_reshaped ** 2, axis=-1)))
    ll_1s = np.log1p(np.mean(np.abs(np.diff(sec_reshaped, axis=-1)), axis=-1))
    mob_1s = np.sqrt(
        np.mean(np.diff(sec_reshaped, axis=-1) ** 2, axis=-1)
        / (np.mean(sec_reshaped ** 2, axis=-1) + 1e-5)
    )

    # 2. STFT de 1 s (Hann 256, sin solapamiento entre segundos) -> 1..40 Hz
    _, _, Z = stft(
        x_clipped,
        fs=fs,
        window="hann",
        nperseg=fs,
        noverlap=0,
        boundary=None,
        padded=False,
        axis=-1,
    )
    power_lin = (np.abs(Z[:, 1:41, :]) ** 2).astype(np.float32)
    P = np.log1p(power_lin)  # (18, 40, dur)

    prob_f = power_lin / (np.sum(power_lin, axis=1, keepdims=True) + 1e-8)
    ent_1s = -np.sum(prob_f * np.log(prob_f + 1e-8), axis=1) / np.log(40.0)
    fs_ratio_1s = np.log1p(power_lin[:, 7:, :].sum(axis=1)) - np.log1p(power_lin[:, :7, :].sum(axis=1))
    bg_ratio_1s = np.log1p(power_lin[:, 12:, :].sum(axis=1)) - np.log1p(power_lin[:, :7, :].sum(axis=1))

    # 3. Agregación causal sobre cada ventana adquirida de 4 segundos (terminada en ends = 4..dur)
    ends = np.arange(4, dur + 1, dtype=np.int32)
    P4 = 0.25 * (P[:, :, 3:] + P[:, :, 2:-1] + P[:, :, 1:-2] + P[:, :, :-3])  # (18, 40, T)
    dP4 = 0.5 * (P[:, :, 3:] + P[:, :, 2:-1]) - 0.5 * (P[:, :, 1:-2] + P[:, :, :-3])

    aux_1s = np.stack([rms_1s, ll_1s, mob_1s, ent_1s, fs_ratio_1s, bg_ratio_1s], axis=1)  # (18, 6, dur)
    aux4 = 0.25 * (aux_1s[:, :, 3:] + aux_1s[:, :, 2:-1] + aux_1s[:, :, 1:-2] + aux_1s[:, :, :-3])
    aux4_z, _ = causal_zscore(aux4, window_sec=win_sec, gap_sec=gap_sec, min_std=min_std)
    aux4_z = np.clip(aux4_z, -6.0, 10.0)

    # 4. Normalización espectral causal por frecuencia frente a los 180 s previos
    Z_spec, mean_base = causal_zscore(P4, window_sec=win_sec, gap_sec=gap_sec, min_std=min_std)
    Z_spec = np.clip(Z_spec, -6.0, 10.0)

    # 5. Cambio en distribución espectral relativa intraventana (invariante a caídas de amplitud global)
    P4_rel = P4 - P4.mean(axis=1, keepdims=True)
    base_rel = mean_base - mean_base.mean(axis=1, keepdims=True)
    dRel = P4_rel - base_rel

    # 8 sub-bandas fisiológicas detalladas entre 1 y 40 Hz
    bins = [(0, 2), (2, 4), (4, 7), (7, 10), (10, 14), (14, 21), (21, 29), (29, 40)]
    feat_list = []
    for a, b in bins:
        feat_list.append(Z_spec[:, a:b, :].mean(axis=1))
        feat_list.append(np.clip(Z_spec[:, a:b, :], 0, 10).mean(axis=1))
        feat_list.append(dRel[:, a:b, :].mean(axis=1))
        feat_list.append(P4[:, a:b, :].mean(axis=1) / 10.0)
        feat_list.append(dP4[:, a:b, :].mean(axis=1))

    # 6. Contraste explícito de desincronización / atenuación ictal: rápido (8-40 Hz) vs lento (1-7 Hz)
    fast_z = Z_spec[:, 7:, :].mean(axis=1)
    slow_z = Z_spec[:, :7, :].mean(axis=1)
    feat_list.append(fast_z - slow_z)
    feat_list.append(np.clip(fast_z, 0, 10) * np.clip(-slow_z, 0, 6))

    for k in range(6):
        feat_list.append(aux4_z[:, k, :])

    # Forma final: (T, 18, 48)
    X = np.stack(feat_list, axis=-1).transpose(1, 0, 2).astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Características causales no finitas detectadas.")
    return X, ends


def extract_record_features(record: dict, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Lee un EDF, aplica el filtro Butterworth causal por bloques y extrae la matriz (T, 18, 48)."""
    dur = int(record["duration"])
    fs = int(config["fs"])
    sos = design_filter(config)
    state = np.zeros((len(sos), 18, 2), dtype=np.float64)
    block_sec = int(config["preprocessing_block_seconds"])
    filtered = np.empty((18, dur * fs), dtype=np.float32)
    for s0 in range(0, dur, block_sec):
        s1 = min(dur, s0 + block_sec)
        raw = causal.read_calibrated(record, s0, s1)
        out, state = sosfilt(sos, raw, axis=1, zi=state)
        filtered[:, s0 * fs : s1 * fs] = out.astype(np.float32)

    channel_sum = filtered.astype(np.float64).sum(axis=1).tolist()
    channel_sq = (filtered.astype(np.float64) ** 2).sum(axis=1).tolist()
    X, ends = extract_features_from_filtered(filtered, config)
    y = labels_at(record, ends)
    stats = dict(
        samples=int(dur * fs),
        windows=int(len(ends)),
        channel_sum=channel_sum,
        channel_sum_squares=channel_sq,
    )
    return X.astype(np.float16), ends, y, stats


class MagnitudePreservingCausalConv(nn.Module):
    """Convolución temporal dilatada estrictamente causal sin normalización destructora de amplitud."""
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(x.transpose(1, 2), (self.pad, 0))
        y = self.conv(y).transpose(1, 2)
        y = self.dropout(F.gelu(y))
        return x + y


class MagnitudePreservingSeizureNet(nn.Module):
    """Red TCN-GRU con agregación espacial invariante a permutaciones y preservación de magnitud Z."""
    def __init__(self, dropout: float = 0.25, config: dict | None = None):
        super().__init__()
        in_features = 48 if config is None else int(config.get("adaptive", {}).get("n_features", 48))
        self.in_scale = nn.Parameter(torch.ones(in_features) * 0.25)
        self.channel_net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.channel_attn = nn.Linear(32, 1)
        self.spatial_proj = nn.Sequential(
            nn.Linear(128, 96),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.tcn1 = MagnitudePreservingCausalConv(96, kernel_size=3, dilation=1, dropout=dropout)
        self.tcn2 = MagnitudePreservingCausalConv(96, kernel_size=3, dilation=2, dropout=dropout)
        self.tcn3 = MagnitudePreservingCausalConv(96, kernel_size=3, dilation=4, dropout=dropout)
        self.tcn4 = MagnitudePreservingCausalConv(96, kernel_size=3, dilation=8, dropout=dropout)
        self.gru = nn.GRU(96, 64, batch_first=True, bidirectional=False)
        self.skip = nn.Linear(96, 64)
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None, return_state: bool = False):
        if x.ndim != 4 or x.shape[2] != 18:
            raise ValueError(f"Se esperaba tensor (B, T, 18, F); se recibió {tuple(x.shape)}.")
        x_s = x * self.in_scale
        h_ch = self.channel_net(x_s)
        mean_p = h_ch.mean(dim=2)
        max_p = h_ch.max(dim=2).values
        top4_p = torch.topk(h_ch, k=4, dim=2).values.mean(dim=2)
        w = torch.softmax(self.channel_attn(h_ch), dim=2)
        attn_p = (h_ch * w).sum(dim=2)
        spatial = torch.cat([mean_p, max_p, top4_p, attn_p], dim=-1)
        s = self.spatial_proj(spatial)
        s = self.tcn4(self.tcn3(self.tcn2(self.tcn1(s))))
        seq, next_state = self.gru(s, state)
        out = seq + self.skip(s)
        logits = self.head(out).squeeze(-1)
        return (logits, next_state) if return_state else logits


def focal_asymmetric_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCE asimétrica con penalización focal sobre falsos positivos de alta confianza."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits).detach()
    neg_weight = 1.0 + 4.0 * (probs ** 2)
    pos_m = targets >= 0.5
    neg_m = ~pos_m
    pos_loss = bce[pos_m].mean() if pos_m.any() else logits.sum() * 0.0
    neg_loss = (bce[neg_m] * neg_weight[neg_m]).mean() if neg_m.any() else logits.sum() * 0.0
    if pos_m.any() and neg_m.any():
        return 0.30 * pos_loss + 0.70 * neg_loss
    return pos_loss + neg_loss


@torch.inference_mode()
def predict_overlapping_causal(
    model: nn.Module,
    X_np: np.ndarray,
    device: torch.device,
    seq_len: int = 24,
    stride: int = 12,
    batch_size: int = 64,
) -> np.ndarray:
    """Inferencia causal en ventanas solapadas de longitud seq_len con avance stride.

    Garantiza que cada segundo emitido t >= seq_len - stride disponga de al menos
    (seq_len - stride) segundos de contexto causal ininterrumpido, y que el estado
    de la GRU nunca supere seq_len pasos (evitando deriva en registros de 1-4 h).
    """
    model.eval()
    T = len(X_np)
    if T == 0:
        return np.zeros(0, dtype=np.float32)
    if T <= seq_len:
        x_t = torch.from_numpy(np.asarray(X_np, dtype=np.float32)[None]).to(device)
        probs = torch.sigmoid(model(x_t)).cpu().numpy()[0]
        if not np.isfinite(probs).all():
            raise ValueError("Predicciones no finitas.")
        return probs

    starts = list(range(0, T - seq_len + 1, stride))
    if starts[-1] != T - seq_len:
        starts.append(T - seq_len)

    windows = np.stack([np.asarray(X_np[s : s + seq_len], dtype=np.float32) for s in starts])
    probs_list = []
    for i in range(0, len(windows), batch_size):
        b = torch.from_numpy(windows[i : i + batch_size]).to(device)
        probs_list.append(torch.sigmoid(model(b)).cpu().numpy())
    probs_w = np.concatenate(probs_list, axis=0)

    out = np.zeros(T, dtype=np.float32)
    out[:seq_len] = probs_w[0]
    keep = seq_len - stride
    for s, pw in zip(starts[1:], probs_w[1:]):
        out[s + keep : s + seq_len] = pw[keep:]
    if not np.isfinite(out).all():
        raise ValueError("Predicciones no finitas.")
    return out


class AdaptiveTrainingSequences:
    """Constructor determinista de secuencias de entrenamiento con minería de negativos difíciles."""
    def __init__(self, entries: list[dict], lookup: dict[str, dict], cache_loader, config: dict):
        self.entries = [e for e in entries if e["role"] == "train"]
        self.lookup = lookup
        self.cache_loader = cache_loader
        self.config = config
        self.seq_len = int(config["sequence_steps"])
        self.seed = int(config["seed"])

        # Seleccionar para el muestreador base todos los EDF ictales de TRAIN + los primeros 2 EDF
        # no ictales por persona (y anclas espaciadas del resto de EDF no ictales)
        by_person_entries: dict[str, list[dict]] = {}
        for e in self.entries:
            by_person_entries.setdefault(e["person"], []).append(e)

        self.pos_by_person: dict[str, list[tuple[str, int]]] = {}
        self.neg_by_person: dict[str, list[tuple[str, int]]] = {}
        self.mining_paths: list[str] = []
        self.samples_list: list[tuple[str, int, int]] = []

        for person in sorted(by_person_entries):
            p_entries = by_person_entries[person]
            ictal_e = [e for e in p_entries if len(lookup[e["path"]]["events"]) > 0]
            non_ictal_e = [e for e in p_entries if len(lookup[e["path"]]["events"]) == 0]
            primary_non_ictal = non_ictal_e[:2]

            for e in ictal_e + primary_non_ictal:
                path = e["path"]
                self.mining_paths.append(path)
                rec = lookup[path]
                dur = int(rec["duration"])
                ends = np.arange(4, dur + 1, dtype=np.int32)
                y = labels_at(rec, ends)
                if len(ends) <= self.seq_len:
                    continue
                for t in range(0, len(ends) - self.seq_len, 1):
                    sub_y = y[t : t + self.seq_len]
                    if sub_y.sum() > 0:
                        self.pos_by_person.setdefault(person, []).append((path, t))
                        self.samples_list.append((path, t, 1))
                    elif t % 2 == 0:
                        self.neg_by_person.setdefault(person, []).append((path, t))
                        self.samples_list.append((path, t, 0))

        self.people = sorted(self.pos_by_person.keys())
        if len(self.people) != 12:
            raise ValueError(f"Se esperaban 12 personas de TRAIN con crisis; se encontraron {len(self.people)}.")

    def next_batch(
        self,
        rng: np.random.Generator,
        cache,
        augment: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Genera un lote balanceado de 36 secuencias (12 positivas + 24 negativas) idéntico al verificador."""
        persons_per_batch = int(self.config["training"]["persons_per_batch"])
        batch_items = []
        for _ in range(persons_per_batch):
            p = str(rng.choice(self.people))
            pos_pool = self.pos_by_person[p]
            neg_pool = self.neg_by_person[p]
            batch_items.append(pos_pool[int(rng.integers(len(pos_pool)))])
            batch_items.append(neg_pool[int(rng.integers(len(neg_pool)))])
            batch_items.append(neg_pool[int(rng.integers(len(neg_pool)))])

        bx, by = [], []
        for path, t0 in batch_items:
            rec = self.lookup[path]
            arr = cache.get(rec)
            seq_x = np.asarray(arr[t0 : t0 + self.seq_len], dtype=np.float32)
            ends = np.arange(4 + t0, 4 + t0 + self.seq_len, dtype=np.int32)
            seq_y = labels_at(rec, ends)
            bx.append(seq_x)
            by.append(seq_y)

        bx_np = np.stack(bx, axis=0)
        by_np = np.stack(by, axis=0)
        if augment:
            scale = rng.uniform(0.85, 1.15, size=(len(batch_items), 1, 1, 1)).astype(np.float32)
            bx_np = bx_np * scale
        return torch.from_numpy(bx_np), torch.from_numpy(by_np)

    def mine_hard_negatives(self, model: nn.Module, cache, device: torch.device) -> int:
        """Minería causal de falsos positivos difíciles sobre registros de TRAIN."""
        model.eval()
        added = 0
        stride = int(self.config["adaptive"]["inference_stride"])
        th_mine = float(self.config["adaptive"]["hard_mining_threshold"])
        for path in self.mining_paths:
            rec = self.lookup[path]
            arr = cache.get(rec)
            dur = int(rec["duration"])
            ends = np.arange(4, dur + 1, dtype=np.int32)
            y = labels_at(rec, ends)
            probs = predict_overlapping_causal(model, arr, device, seq_len=self.seq_len, stride=stride)
            fp_idx = np.where((probs > th_mine) & (y == 0))[0]
            for idx_fp in fp_idx[::2]:
                t0_fp = max(0, min(len(ends) - self.seq_len, int(idx_fp) - self.seq_len + 2))
                if y[t0_fp : t0_fp + self.seq_len].sum() == 0:
                    self.neg_by_person[rec["person"]].append((path, t0_fp))
                    added += 1
        return added
