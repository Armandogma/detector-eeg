"""Ensayo aislado 12/4/4: TCN-GRU causal-adaptativo con preservación de magnitud Z.

Ejecutable únicamente desde su paquete aislado en Google Colab o entorno de verificación.
No modifica fuentes ni cachés de experimentos históricos.
"""
import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import zipfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import psutil
from scipy.signal import butter, sosfilt
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.nn import functional as F

import dataset_causal as causal
from ajustes_modelo import causal_average
from eventos_relativos import event_row, strict_rows, sample_metrics, ranking_metrics
from evaluador_continuo import aggregate, bootstrap_people
from rutas import REPO
from modelo_adaptativo import (
    MagnitudePreservingSeizureNet,
    AdaptiveTrainingSequences as TrainingWindows,
    extract_features_from_filtered,
    focal_asymmetric_bce,
    labels_at,
    predict_overlapping_causal,
)

OUT = REPO / "entrenamiento"
CACHE = REPO / "preparados"
GIB = 1024 ** 3


def RecortesModel(dropout=0.25, config=None):
    return MagnitudePreservingSeizureNet(
        dropout=dropout,
        config=config or read(REPO / "codigo/configuracion_adaptativa.json"),
    )


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def save_torch(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + ".partial")
    torch.save(payload, temp)
    os.replace(temp, path)


def pack_check():
    meta = read(REPO / "paquete_manifest.json")
    if meta["family"] != "tcn_gru_espectral_adaptativo_v1":
        raise ValueError("Este programa solo se ejecuta dentro de su paquete aislado.")
    for name, expected in meta["files"].items():
        if digest(REPO / name) != expected:
            raise ValueError("Fuente del paquete modificada: " + name)
    return meta


def versions():
    return dict(
        python=f"{sys.version_info.major}.{sys.version_info.minor}",
        **{
            p: importlib.metadata.version(p)
            for p in ("torch", "numpy", "scipy", "scikit-learn", "psutil", "matplotlib")
        },
    )


def inputs():
    manifest = read(REPO / "analisis/manifiesto_dataset.json")
    cohort = read(REPO / "analisis/cohorte_pediatrica.json")
    split = read(REPO / "analisis/particion_rf_fija.json")
    paths = set(cohort["eligible_paths"])
    records = [r for r in manifest["records"] if r["path"] in paths]
    if len(records) != 590 or sum(len(r["events"]) for r in records) != 168:
        raise ValueError("Cohorte distinta de 590 EDF y 168 crisis.")
    groups = [set(split[k]) for k in ("train", "validation", "test")]
    if [len(g) for g in groups] != [12, 4, 4] or any(
        groups[i] & groups[j] for i in range(3) for j in range(i)
    ):
        raise ValueError("Las personas deben estar separadas 12/4/4.")
    if set.union(*groups) != set(cohort["people"]):
        raise ValueError("Personas fuera de la partición.")
    for key, people, events in zip(("train", "validation", "test"), groups, (113, 24, 31)):
        subset = [r for r in records if r["person"] in people]
        if len(subset) != split["counts"][key]["edf"] or sum(len(r["events"]) for r in subset) != events:
            raise ValueError("Conteo incorrecto: " + key)
    if any(r["person"] != "chb01" for r in records if r["case"] in ("chb01", "chb21")):
        raise ValueError("chb01/chb21 deben ser una sola persona.")
    return records, split


def intervals_for_record(record, config, training, is_primary=True):
    duration = int(record["duration"])
    if not training or is_primary or len(record["events"]) > 0:
        return [dict(start=0, stop=duration, kind="continuous", offset_samples=0)]
    span = min(duration, 120)
    return [dict(start=0, stop=span, kind="far_negative", offset_samples=0)]


def build_index(records, split, config):
    primary_paths = set()
    for p in sorted(split["train"]):
        p_recs = [r for r in records if r["person"] == p]
        for r in p_recs:
            if len(r["events"]) > 0:
                primary_paths.add(r["path"])
        for r in [r for r in p_recs if len(r["events"]) == 0][:2]:
            primary_paths.add(r["path"])

    result = []
    for record in records:
        role = next(k for k in ("train", "validation", "test") if record["person"] in split[k])
        if role == "test":
            continue
        is_prim = (role != "train") or (record["path"] in primary_paths)
        result.append(
            dict(
                path=record["path"],
                person=record["person"],
                role=role,
                segments=intervals_for_record(record, config, role == "train", is_prim),
            )
        )
    return result


def check_edf(record):
    source = causal.DATASET / record["path"]
    if not source.is_file() or source.stat().st_size != record["file_bytes"]:
        raise ValueError("EDF ausente o tamaño diferente: " + str(source))
    with source.open("rb") as stream:
        if hashlib.sha256(stream.read(record["header"]["hbytes"])).hexdigest() != record["header_sha256"]:
            raise ValueError("Cabecera diferente: " + record["path"])
    if record.get("annotation_sha256") and digest(Path(str(source) + ".seizures")) != record["annotation_sha256"]:
        raise ValueError("Anotación diferente: " + record["path"])
    if digest(source.parent / (record["case"] + "-summary.txt")) != record["summary_sha256"]:
        raise ValueError("Resumen diferente: " + record["path"])


def cache_path(record):
    return CACHE / (causal.identifier(record) + ".npy")


def prepare_record(record, segments, config, reader=None):
    """Filtra causalmente el registro y almacena la matriz (T, 18, 48) en float16."""
    reader = reader or causal.read_calibrated
    CACHE.mkdir(parents=True, exist_ok=True)
    target = cache_path(record)
    meta_path = target.with_suffix(".json")
    contract = dict(
        source=record["path"],
        header=record["header_sha256"],
        segments=segments,
        adaptive=config["adaptive"],
        filter={
            k: config[k]
            for k in ("fs", "filter_design_order", "filter_low_hz", "filter_high_hz", "filter_initial_state")
        },
    )
    if target.exists() and meta_path.exists():
        meta = read(meta_path)
        if meta["contract"] != contract or digest(target) != meta["sha256"]:
            raise ValueError("Caché modificada o incompatible: " + str(target))
        return meta

    stop_sec = max(int(b["stop"]) for b in segments)
    fs = int(config["fs"])
    sos = butter(
        config["filter_design_order"],
        [config["filter_low_hz"], config["filter_high_hz"]],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    state = np.zeros((len(sos), 18, 2), dtype=np.float64)
    full_filtered = np.empty((18, stop_sec * fs), dtype=np.float32)
    n = 0
    total = np.zeros(18, dtype=np.float64)
    square = np.zeros(18, dtype=np.float64)
    block_sec = int(config["preprocessing_block_seconds"])

    for s0 in range(0, stop_sec, block_sec):
        s1 = min(s0 + block_sec, stop_sec)
        filtered, state = sosfilt(sos, reader(record, s0, s1), axis=1, zi=state)
        if filtered.shape != (18, (s1 - s0) * fs) or not np.isfinite(filtered).all():
            raise ValueError("Señales no finitas o dimensiones incorrectas.")
        n += filtered.shape[1]
        total += filtered.sum(axis=1)
        square += (filtered ** 2).sum(axis=1)
        full_filtered[:, s0 * fs : s1 * fs] = filtered.astype(np.float32)

    X_feats, ends = extract_features_from_filtered(full_filtered, config)
    X_f16 = X_feats.astype(np.float16)

    temp = target.with_name(target.stem + ".partial.npy")
    np.save(temp, X_f16, allow_pickle=False)
    os.replace(temp, target)

    meta = dict(
        contract=contract,
        sha256=digest(target),
        sample_count=int(n),
        sum=total.tolist(),
        sum_squares=square.tolist(),
        saved_windows=int(len(ends)),
        n_features=int(X_f16.shape[-1]),
        dtype="float16",
        units="causal_zscore",
    )
    atomic_json(meta_path, meta)
    return meta


def training_stats(entries, metadata):
    train = [e for e in entries if e["role"] == "train"]
    if not train:
        raise ValueError("Sin entrenamiento.")
    n = sum(metadata[e["path"]]["sample_count"] for e in train)
    mean = sum((np.array(metadata[e["path"]]["sum"]) for e in train), np.zeros(18)) / n
    second = sum((np.array(metadata[e["path"]]["sum_squares"]) for e in train), np.zeros(18)) / n
    std = np.sqrt(np.maximum(second - mean ** 2, 0))
    if np.any(std <= 0):
        raise ValueError("Normalización degenerada.")
    return dict(
        mean=mean.tolist(),
        std=std.tolist(),
        sample_count=int(n),
        fitted_records=[e["path"] for e in train],
        scope="Causal filter moments across prepared TRAIN segments; features use per-record 180s past z-score.",
    )


class SignalCache:
    def __init__(self, budget_gib=16, reserve_gib=4):
        memory = psutil.virtual_memory()
        self.budget = int(max(0, min(budget_gib * GIB, 0.75 * memory.total, memory.available - reserve_gib * GIB)))
        self.reserve = int(reserve_gib * GIB)
        self.arrays = OrderedDict()
        self.maps = OrderedDict()
        self.used = 0

    def get(self, record):
        key = record["path"]
        path = cache_path(record)
        if key in self.arrays:
            return self.arrays[key]
        size = path.stat().st_size
        while self.arrays and psutil.virtual_memory().available < self.reserve:
            _, old = self.arrays.popitem(last=False)
            self.used -= old.nbytes
        if self.used + size <= self.budget and psutil.virtual_memory().available >= self.reserve + size:
            array = np.load(path, allow_pickle=False)
            self.arrays[key] = array
            self.used += array.nbytes
            return array
        if key in self.maps:
            value = self.maps.pop(key)
        else:
            while len(self.maps) >= 16:
                _, old = self.maps.popitem(last=False)
                old._mmap.close()
            value = np.load(path, mmap_mode="r", allow_pickle=False)
        self.maps[key] = value
        return value

    def close(self):
        self.arrays.clear()
        self.used = 0
        for array in self.maps.values():
            if getattr(array, "_mmap", None) is not None:
                array._mmap.close()
        self.maps.clear()


def deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resource_info():
    memory = psutil.virtual_memory()
    return dict(
        ram_total_gib=memory.total / GIB,
        ram_available_gib=memory.available / GIB,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        vram_gib=torch.cuda.get_device_properties(0).total_memory / GIB if torch.cuda.is_available() else None,
    )


@torch.inference_mode()
def predict(model, records, stats, variant, config, device, cache=None):
    model.eval()
    output = []
    owned = cache is None
    if owned:
        cache = SignalCache(config["performance"]["ram_cache_gib"], config["performance"]["reserve_ram_gib"])
    seq_len = int(config["sequence_steps"])
    stride = int(config["adaptive"]["inference_stride"])
    bsize = int(config["inference_batch_size"])
    try:
        for i, record in enumerate(records, 1):
            ends, labels, _ = causal.record_index(record, config)
            probabilities = predict_overlapping_causal(
                model, cache.get(record), device, seq_len=seq_len, stride=stride, batch_size=bsize
            )
            output.append(dict(ends=ends, labels=labels, probabilities=probabilities))
            if i % 25 == 0 or i == len(records):
                print(f"Inferencia {variant}: {i}/{len(records)} EDF", flush=True)
    finally:
        if owned:
            cache.close()
    return output


def save_predictions(path, records, predictions):
    payload = {}
    for i, (record, prediction) in enumerate(zip(records, predictions)):
        payload[f"path_{i}"] = np.array(record["path"])
        for key, value in prediction.items():
            payload[f"{key}_{i}"] = value
    temp = Path(path).with_name(Path(path).name + ".partial")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temp, path)


def load_predictions(path, records, config):
    with np.load(path, allow_pickle=False) as packed:
        expected_keys = {f"{key}_{i}" for i in range(len(records)) for key in ("path", "ends", "labels", "probabilities")}
        if set(packed.files) != expected_keys:
            raise ValueError("Inventario de predicciones distinto.")
        result = []
        for i, record in enumerate(records):
            ends, labels, _ = causal.record_index(record, config)
            probabilities = packed[f"probabilities_{i}"]
            if (
                str(packed[f"path_{i}"]) != record["path"]
                or not np.array_equal(ends, packed[f"ends_{i}"])
                or not np.array_equal(labels, packed[f"labels_{i}"])
                or probabilities.shape != ends.shape
                or not np.isfinite(probabilities).all()
                or np.any((probabilities < 0) | (probabilities > 1))
            ):
                raise ValueError("Predicciones/etiquetas inválidas: " + record["path"])
            result.append(dict(ends=ends, labels=labels, probabilities=probabilities))
    return result


def point(records, predictions, threshold, rule):
    rows = [event_row(r, p, threshold, rule) for r, p in zip(records, predictions)]
    persons = sorted({r["person"] for r in rows})
    by_person = {p: aggregate([r for r in rows if r["person"] == p]) for p in persons}
    strict = strict_rows(rows)
    return dict(
        threshold=float(threshold),
        rule=rule,
        total=aggregate(rows),
        by_person=by_person,
        macro_sensitivity=float(np.mean([p["sensitivity"] for p in by_person.values()])),
        by_record=rows,
        onset_only_total=aggregate(strict),
        onset_only_by_person={p: aggregate([r for r in strict if r["person"] == p]) for p in persons},
        nonictal_alarm_seconds=sum(r["nonictal_alarm_seconds"] for r in rows),
        longest_alarm_seconds=max([r["longest_alarm_seconds"] for r in rows] or [0]),
        scoring="Any positive overlap, one-to-one, no tolerance, no merges or event splits; strict onset also reported.",
    )


def choose_alarm(records, predictions, config):
    compact = []
    best = None
    best_key = None
    for rule in config["rules"]:
        thresholds = set(config["thresholds"])
        smooth = [causal_average(p["probabilities"], rule["smooth_seconds"]) for p in predictions]
        for person in sorted({r["person"] for r in records}):
            labels = np.concatenate([p["labels"] for r, p in zip(records, predictions) if r["person"] == person])
            scores = np.concatenate([s for r, s in zip(records, smooth) if r["person"] == person])
            for label in (0, 1):
                subset = scores[labels == label]
                if len(subset):
                    thresholds.update(map(float, np.quantile(subset, config["validation_threshold_quantiles"])))
        for threshold in sorted(thresholds):
            candidate = point(records, predictions, threshold, rule)
            compact.append({k: v for k, v in candidate.items() if k != "by_record"})
            total = candidate["total"]
            if (
                total["false_alarms_per_hour"] <= config["validation_false_alarms_budget_per_hour"]
                and total["time_under_alarm"] <= config["validation_max_alarm_time_fraction"]
            ):
                key = (
                    acceptable(candidate, config),
                    candidate["macro_sensitivity"],
                    total["sensitivity"],
                    -total["false_alarms_per_hour"],
                    -total["time_under_alarm"],
                    candidate["threshold"],
                )
                if best_key is None or key > best_key:
                    best = candidate
                    best_key = key
    if best is None:
        raise ValueError("Falta candidato sin alarmas.")
    return best, compact


def acceptable(point_value, config):
    total = point_value["total"]
    return (
        total["sensitivity"] >= config["validation_min_pooled_sensitivity"]
        and point_value["macro_sensitivity"] >= config["validation_min_macro_sensitivity"]
        and total["false_alarms_per_hour"] <= config["validation_false_alarms_budget_per_hour"]
        and total["time_under_alarm"] <= config["validation_max_alarm_time_fraction"]
        and (not config["validation_no_zero_person"] or all(p["detected"] > 0 for p in point_value["by_person"].values()))
    )


def decide(points, config):
    accepted = {key: acceptable(value, config) for key, value in points.items()}
    selected = "adaptativo" if accepted["adaptativo"] else None
    return dict(
        status="accepted" if selected else "rejected",
        selected=selected,
        eligible=accepted,
        reason="Decisión exclusiva de validación. Si se rechaza no ejecutar prueba.",
        prior_test_exposure=True,
        clinical_validation=False,
    )


def macro_ap(records, predictions):
    values = []
    for person in sorted({r["person"] for r in records}):
        labels = np.concatenate([p["labels"] for r, p in zip(records, predictions) if r["person"] == person])
        probs = np.concatenate([p["probabilities"] for r, p in zip(records, predictions) if r["person"] == person])
        if len(np.unique(labels)) != 2:
            raise ValueError("Validación debe contener ambas clases por persona.")
        values.append(float(average_precision_score(labels, probs)))
    return float(np.mean(values))


def backup(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    files = sorted(
        p for p in OUT.iterdir() if p.is_file() and ".partial" not in p.name and p.suffix not in (".lock", ".log")
    )
    hashes = {p.name: digest(p) for p in files}
    temp = destination / "avance.partial.zip"
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_STORED) as archive:
        for file in files:
            archive.write(file, "entrenamiento/" + file.name)
        archive.writestr("respaldo.json", json.dumps(dict(family="adaptativo_v1", files=hashes)))
    with zipfile.ZipFile(temp) as archive:
        if archive.testzip():
            raise ValueError("Respaldo no íntegro.")
    if (destination / "avance.zip").exists():
        os.replace(destination / "avance.zip", destination / "avance_anterior.zip")
    os.replace(temp, destination / "avance.zip")


def restore(destination):
    for name in ("avance.zip", "avance_anterior.zip"):
        path = Path(destination) / name
        if not path.exists():
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                if archive.testzip():
                    raise ValueError("CRC de respaldo no válido.")
                meta = json.loads(archive.read("respaldo.json"))
                if meta["family"] != "adaptativo_v1":
                    raise ValueError("Respaldo de otra familia.")
                if set(archive.namelist()) != {"respaldo.json"} | {"entrenamiento/" + n for n in meta["files"]}:
                    raise ValueError("Inventario de respaldo diferente.")
                payload = {}
                for n, h in meta["files"].items():
                    if Path(n).name != n or "\\" in n or ":" in n:
                        raise ValueError("Ruta de respaldo inválida.")
                    raw = archive.read("entrenamiento/" + n)
                    if hashlib.sha256(raw).hexdigest() != h:
                        raise ValueError("Huella de respaldo diferente.")
                    payload[n] = raw
            extras = [p for p in OUT.iterdir() if p.is_file() and p.name not in payload and p.suffix != ".lock"]
            for file in extras:
                folder = REPO / "interrumpido"
                folder.mkdir(exist_ok=True)
                target = folder / file.name
                number = 1
                while target.exists():
                    target = folder / (file.name + f".{number}")
                    number += 1
                os.replace(file, target)
            for n, raw in payload.items():
                temp = OUT / (n + ".partial")
                temp.write_bytes(raw)
                os.replace(temp, OUT / n)
            print("Respaldo recuperado:", name, flush=True)
            return True
        except (ValueError, KeyError, zipfile.BadZipFile) as exc:
            print("Respaldo no recuperable:", name, str(exc), flush=True)
    if any((Path(destination) / n).exists() for n in ("avance.zip", "avance_anterior.zip")):
        raise ValueError("Conservar Drive: no hay un respaldo íntegro recuperable.")
    return False


@contextlib.contextmanager
def process_lock():
    path = OUT / "ejecucion.lock"
    stream = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        stream.close()


def freeze(config, split, entries, package):
    expected = dict(config=config, split=split, index=entries, package=package, versions=versions(), gpu=resource_info()["gpu"])
    path = OUT / "protocolo.json"
    if path.exists():
        frozen = read(path)
        for key, value in expected.items():
            if frozen[key] != value:
                raise ValueError("Cambió el protocolo/entorno; no mezclar avances: " + key)
    else:
        atomic_json(
            path,
            dict(
                **expected,
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                prior_test_exposure=True,
                clinical_validation=False,
            ),
        )
    return digest(path)


def bind_preparation(protocol_sha, create=False):
    payload = dict(
        protocol_sha=protocol_sha,
        files={name: digest(OUT / name) for name in ("normalizacion.json", "cache_manifest.json", "recortes.json", "recortes.csv")},
    )
    path = OUT / "datos_congelados.json"
    if path.exists():
        if read(path) != payload:
            raise ValueError("Cambió la normalización, la selección de recortes o su caché congelada.")
    elif create:
        atomic_json(path, payload)
    else:
        raise ValueError("Preparación no congelada. Ejecuta prepare.")
    return digest(path)


def prepare_entries(entries, lookup, config):
    required = sum(
        sum(max(1, (s["stop"] - s["start"])) * 18 * 48 * 2 for s in e["segments"])
        for e in entries
        if not cache_path(lookup[e["path"]]).exists()
    )
    if shutil.disk_usage(REPO).free < required + 2 * GIB:
        raise ValueError(f"Disco insuficiente: faltan {required/GIB:.2f} GiB de caché y 2 GiB de margen.")

    workers = int(config.get("performance", {}).get("prepare_workers", 6))
    for entry in entries:
        check_edf(lookup[entry["path"]])

    def _work(item):
        idx, entry = item
        rec = lookup[entry["path"]]
        meta = prepare_record(rec, entry["segments"], config)
        return idx, entry["path"], meta

    result = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for idx, path, meta in ex.map(_work, enumerate(entries, 1)):
            result[path] = meta
            if idx % 25 == 0 or idx == len(entries):
                print(f"Preparación {idx}/{len(entries)} EDF", flush=True)
    return result


def validate_caches(entries, lookup, expected):
    for entry in entries:
        path = cache_path(lookup[entry["path"]])
        if not path.exists() or digest(path) != expected[entry["path"]]["sha256"]:
            raise ValueError("Caché ausente o modificada. Ejecuta prepare para recuperarla: " + entry["path"])


def train_variant(variant, data, validation, stats, config, device, ram_gib, drive, protocol_sha):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    model = RecortesModel(config["training"]["dropout"], config).to(device)
    t = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    rng = np.random.default_rng(config["seed"])

    checkpoint = OUT / (variant + "_ultimo.pt")
    start = 0
    history = []
    best_key = None
    best_ap = -1.0
    best_epoch = 0
    best_model = None

    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["protocol_sha"] != protocol_sha or saved["variant"] != variant:
            raise ValueError("Checkpoint de otro protocolo.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        start = saved["epoch"]
        history = saved["history"]
        best_ap = saved["best_ap"]
        best_epoch = saved["best_epoch"]
        best_model = saved["best_model"]
        best_key = tuple(saved["best_key"]) if saved.get("best_key") is not None else None
        rng.bit_generator.state = saved["rng_state"]
        data.neg_by_person = saved["neg_by_person"]

    cache = SignalCache(ram_gib, config["performance"]["reserve_ram_gib"])
    print(f"{variant}: desde época {start + 1}, caché RAM máxima {cache.budget / GIB:.1f} GiB", flush=True)
    steps = int(t["batches_per_epoch"])
    mine_epochs = set(config["adaptive"]["hard_mining_epochs"])

    try:
        for epoch in range(start + 1, t["max_epochs"] + 1):
            model.train()
            losses = []
            tick = time.perf_counter()
            for step in range(1, steps + 1):
                bx, by = data.next_batch(rng, cache, augment=True)
                bx = bx.to(device)
                by = by.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(bx)
                loss = focal_asymmetric_bce(logits, by)
                if not torch.isfinite(loss):
                    raise ValueError("Pérdida no finita.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip"])
                optimizer.step()
                losses.append(float(loss.detach()))
                if step % 50 == 0 or step == steps:
                    print(f"{variant} época {epoch}/{t['max_epochs']}: lote {step}/{steps}, pérdida {np.mean(losses):.4f}", flush=True)

            mined = 0
            if epoch in mine_epochs:
                mined = data.mine_hard_negatives(model, cache, device)
                print(f"{variant} época {epoch}: minados {mined} negativos difíciles en TRAIN", flush=True)

            predictions = predict(model, validation, stats, variant, config, device, cache=cache)
            ap = macro_ap(validation, predictions)

            if epoch >= t["selection_start_epoch"]:
                cand, _ = choose_alarm(validation, predictions, config)
                is_acc = int(acceptable(cand, config))
                cur_key = (
                    is_acc,
                    round(cand["macro_sensitivity"], 6),
                    round(cand["total"]["sensitivity"], 6),
                    round(ap, 6),
                    -round(cand["total"]["false_alarms_per_hour"], 6),
                )
                if best_key is None or cur_key > best_key:
                    best_key = cur_key
                    best_ap = ap
                    best_epoch = epoch
                    best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            history.append(
                dict(
                    epoch=epoch,
                    loss=float(np.mean(losses)),
                    bce=float(np.mean(losses)),
                    validation_macro_ap=ap,
                    mined_hard_negatives=int(mined),
                    seconds=time.perf_counter() - tick,
                )
            )
            payload = dict(
                protocol_sha=protocol_sha,
                variant=variant,
                epoch=epoch,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                history=history,
                best_ap=best_ap,
                best_epoch=best_epoch,
                best_model=best_model,
                best_key=list(best_key) if best_key is not None else None,
                rng_state=rng.bit_generator.state,
                neg_by_person=data.neg_by_person,
            )
            save_torch(checkpoint, payload)
            atomic_json(OUT / (variant + "_historia.json"), history)
            atomic_json(
                OUT / "estado.json",
                dict(
                    stage="training",
                    variant=variant,
                    completed_epoch=epoch,
                    max_epochs=t["max_epochs"],
                    best_epoch=best_epoch,
                    validation_macro_ap=ap,
                ),
            )
            write_rows(OUT / "historia.csv", history)
            backup(drive)
            print(f"{variant}: época {epoch} guardada. AP media validación {ap:.4f}; mejor época {best_epoch}.", flush=True)
    finally:
        cache.close()

    if best_model is None:
        raise ValueError("No hay una época seleccionable.")
    model.load_state_dict(best_model)
    save_torch(OUT / (variant + "_mejor.pt"), dict(model=best_model, epoch=best_epoch, variant=variant, protocol_sha=protocol_sha))
    predictions = predict(model, validation, stats, variant, config, device)
    path = OUT / (variant + "_validacion.npz")
    save_predictions(path, validation, predictions)
    verified = load_predictions(path, validation, config)
    chosen, curve = choose_alarm(validation, verified, config)
    if chosen["total"]["events"] != sum(len(r["events"]) for r in validation) or chosen["total"]["partial_events"]:
        raise ValueError("La validación debe conservar todas sus crisis completas.")
    chosen.update(
        best_epoch=best_epoch,
        validation_macro_ap=best_ap,
        weights_sha256=digest(OUT / (variant + "_mejor.pt")),
        predictions_sha256=digest(path),
        protocol_sha=protocol_sha,
    )
    chosen["sample_metrics"] = sample_metrics(validation, verified, chosen["threshold"], chosen["rule"])
    chosen["ranking_metrics_by_person"] = ranking_metrics(validation, verified)
    atomic_json(OUT / (variant + "_seleccion.json"), chosen)
    atomic_json(OUT / (variant + "_curva.json"), curve)
    backup(drive)
    return chosen


def pilot(data, stats, config, device, protocol_sha):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    model = RecortesModel(0.0, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["pilot"]["learning_rate"], weight_decay=0.0)
    cache = SignalCache(4, 2)
    rng = np.random.default_rng(config["seed"])
    history = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        x, y = data.next_batch(rng, cache, augment=False)
        x = x.to(device)
        y = y.to(device)
        model.eval()
        with torch.no_grad():
            initial = float(focal_asymmetric_bce(model(x), y))
        tick = time.perf_counter()
        for step in range(1, config["pilot"]["steps"] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = focal_asymmetric_bce(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Pérdida no finita en el piloto.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            optimizer.step()
            history.append(dict(step=step, loss=float(loss.detach())))
            if step % 15 == 0:
                print("Piloto TRAIN:", step, "/", config["pilot"]["steps"], "pérdida", history[-1]["loss"], flush=True)
        model.eval()
        with torch.no_grad():
            final = float(focal_asymmetric_bce(model(x), y))
        if device.type == "cuda":
            torch.cuda.synchronize()
    finally:
        cache.close()

    passed = final <= initial * config["pilot"]["maximum_final_loss_ratio"]
    result = dict(
        protocol_sha=protocol_sha,
        passed=passed,
        initial_loss=initial,
        final_loss=final,
        required_final_ratio=config["pilot"]["maximum_final_loss_ratio"],
        history=history,
        seconds=time.perf_counter() - tick,
        resources=resource_info(),
        peak_tensor_vram_mib=torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else None,
        not_a_model_result=True,
        weights_discarded=True,
        scope="Fixed TRAIN batch only. Sanity check of learning capacity; no validation or test. Not generalization.",
    )
    atomic_json(OUT / "piloto.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2), flush=True)
    return result


def write_rows(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_alarm_audit(result, artificial=False):
    import matplotlib.pyplot as plt
    people = sorted(result["by_person"])
    x = np.arange(len(people))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    overlap = [result["by_person"][p]["detected"] for p in people]
    onset = [result["onset_only_by_person"][p]["detected"] for p in people]
    axes[0].bar(x - 0.18, overlap, 0.36, label="Solapamiento uno a uno", color="#147D92")
    axes[0].bar(x + 0.18, onset, 0.36, label="Inicio dentro de crisis", color="#777777")
    axes[0].set(xticks=x, xticklabels=people, ylabel="Crisis detectadas", title="Mismas puntuaciones y regla")
    axes[0].legend(fontsize=9)
    axes[1].bar(people, [100 * result["by_person"][p]["time_under_alarm"] for p in people], color="#A55A32")
    axes[1].set(ylabel="Tiempo bajo alarma (%)", title="Todo el tiempo marcado permanece contado")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    prefix = "DATOS ARTIFICIALES · " if artificial else ""
    fig.suptitle(prefix + "Auditoría de alarmas. Duración máxima: " + str(round(result["longest_alarm_seconds"], 1)) + " s")
    fig.savefig(OUT / "alarmas.png", dpi=160)
    plt.close(fig)


def plot_training_signal(data, config, artificial=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    first_person = data.people[0]
    path, t0 = data.pos_by_person[first_person][0]
    record = data.lookup[path]
    arr = np.load(cache_path(record), mmap_mode="r")
    seq_len = int(config["sequence_steps"])
    feats = np.asarray(arr[t0 : t0 + seq_len], dtype=np.float32)
    if getattr(arr, "_mmap", None) is not None:
        arr._mmap.close()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    im0 = axes[0].imshow(feats[:, 0, :40].T, origin="lower", aspect="auto", cmap="coolwarm")
    axes[0].set(xlabel="Paso en secuencia (s)", ylabel="Descriptor espectral (0..39)", title="Canal FP1-F7 · Puntajes Z y dRel")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(feats[:, :, 40].T, origin="lower", aspect="auto", cmap="viridis")
    axes[1].set(xlabel="Paso en secuencia (s)", ylabel="Derivación (1..18)", title="Contraste rápido-vs-lento (8-40 vs 1-7 Hz)")
    fig.colorbar(im1, ax=axes[1])
    prefix = "DATOS ARTIFICIALES · " if artificial else ""
    fig.suptitle(prefix + f"TRAIN · {path} · secuencia iniciada en ventana {t0}. No son predicciones.")
    fig.savefig(OUT / "senales.png", dpi=160)
    plt.close(fig)


def figures(config, decision, lookup, stats, artificial=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    prefix = "DATOS ARTIFICIALES · " if artificial else ""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for variant, color in (("adaptativo", "#147D92"),):
        path = OUT / (variant + "_historia.json")
        if path.exists():
            history = read(path)
            axes[0].plot([x["epoch"] for x in history], [x["loss"] for x in history], label="adaptativo", color=color, marker="o")
            axes[1].plot(
                [x["epoch"] for x in history],
                [x["validation_macro_ap"] for x in history],
                label="adaptativo",
                color=color,
                marker="o",
            )
    axes[0].set(xlabel="Época completa", ylabel="Pérdida de entrenamiento")
    axes[1].set(xlabel="Época completa", ylabel="AP media por persona · validación", ylim=(0, 1.02))
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle(prefix + "Aprendizaje; AP no es sensibilidad por crisis")
    fig.tight_layout()
    fig.savefig(OUT / "aprendizaje.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant, color in (("adaptativo", "#147D92"),):
        curve = read(OUT / (variant + "_curva.json"))
        ax.scatter(
            [p["total"]["false_alarms_per_hour"] for p in curve],
            [100 * p["total"]["sensitivity"] for p in curve],
            s=14,
            alpha=0.5,
            label=variant,
            color=color,
        )
    ax.axvline(config["validation_false_alarms_budget_per_hour"], color="#A55A32", linestyle="--", label="Límite fijado")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set(
        xlabel="Falsas alarmas por hora no ictal",
        ylabel="Crisis detectadas (%)",
        ylim=(-3, 103),
        xlim=(0, None),
        title=prefix + "Solo validación · todas las reglas ensayadas",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "validacion.png", dpi=180)
    plt.close(fig)

    if (OUT / "evaluacion.json").exists():
        result = read(OUT / "evaluacion.json")
        label = "PRUEBA · " + decision["selected"]
    else:
        options = {v: read(OUT / (v + "_seleccion.json")) for v in config["variants"]}
        key = decision["selected"] or max(options, key=lambda v: options[v]["macro_sensitivity"])
        result = options[key]
        label = "VALIDACIÓN · " + key + " · no es prueba"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    persons = sorted(result["by_person"])
    values = [result["by_person"][p] for p in persons]
    bars = ax.bar(persons, [100 * v["sensitivity"] for v in values], color="#147D92")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{value['detected']}/{value['events']}",
            ha="center",
        )
    ax.set(ylim=(0, 112), ylabel="Sensibilidad por crisis (%)", title=prefix + label)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "personas.png", dpi=180)
    plt.close(fig)

    delays = [d for row in result["by_record"] for d in row["delays"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if delays:
        ax.hist(delays, bins=min(15, max(3, len(delays))), color="#777777", edgecolor="white")
    else:
        ax.text(0.5, 0.5, "Sin crisis detectadas; demora no estimable", ha="center", transform=ax.transAxes)
    ax.set(
        xlabel="Demora desde el inicio anotado (s)",
        ylabel="Crisis detectadas",
        title=prefix + label + "\nDemoras solo entre crisis detectadas",
    )
    fig.tight_layout()
    fig.savefig(OUT / "demoras.png", dpi=180)
    plt.close(fig)
    plot_alarm_audit(result, artificial)
    return result


def evaluate_test(records, split, lookup, config, stats, decision, drive, protocol_sha, device):
    if decision["status"] != "accepted":
        print("VALIDACIÓN RECHAZADA: no se preparará ni evaluará prueba.", flush=True)
        return
    variant = decision["selected"]
    selection = read(OUT / (variant + "_seleccion.json"))
    if digest(OUT / (variant + "_seleccion.json")) != decision["selection_sha256"]:
        raise ValueError("Selección modificada después de fijar validación.")
    if digest(OUT / (variant + "_mejor.pt")) != selection["weights_sha256"]:
        raise ValueError("Pesos distintos de los elegidos.")
    test = [r for r in records if r["person"] in split["test"]]
    if (OUT / "evaluacion.json").exists():
        result = read(OUT / "evaluacion.json")
        if (
            result["decision_sha256"] != digest(OUT / "decision_validacion.json")
            or result["predictions_sha256"] != digest(OUT / "prueba.npz")
        ):
            raise ValueError("Cambió la prueba calculada.")
        predictions = load_predictions(OUT / "prueba.npz", test, config)
    else:
        entries = [dict(path=r["path"], segments=intervals_for_record(r, config, False, True)) for r in test]
        prepare_entries(entries, lookup, config)
        saved = torch.load(OUT / (variant + "_mejor.pt"), map_location="cpu", weights_only=True)
        if saved["protocol_sha"] != protocol_sha:
            raise ValueError("Pesos de otro protocolo.")
        model = RecortesModel(config["training"]["dropout"], config).to(device)
        model.load_state_dict(saved["model"])
        predictions = predict(model, test, stats, variant, config, device)
        save_predictions(OUT / "prueba.npz", test, predictions)
        predictions = load_predictions(OUT / "prueba.npz", test, config)
    result = point(test, predictions, selection["threshold"], selection["rule"])
    if result["total"]["events"] != 31 or result["total"]["partial_events"]:
        raise ValueError("La prueba debe conservar las 31 crisis completas.")
    result.update(
        variant=variant,
        decision_sha256=digest(OUT / "decision_validacion.json"),
        predictions_sha256=digest(OUT / "prueba.npz"),
        weights_sha256=selection["weights_sha256"],
        prior_test_exposure=True,
        clinical_validation=False,
        coverage_denominator="All 138 eligible test EDF; technical cohort exclusions are separately declared.",
        confidence_intervals=bootstrap_people(result["by_record"], 1000, config["seed"]),
    )
    result["sample_metrics"] = sample_metrics(test, predictions, selection["threshold"], selection["rule"])
    result["ranking_metrics_by_person"] = ranking_metrics(test, predictions)
    atomic_json(OUT / "evaluacion.json", result)
    write_rows(
        OUT / "personas.csv",
        [
            dict(
                person=p,
                events=v["events"],
                detected=v["detected"],
                sensitivity=v["sensitivity"],
                false_alarms_per_hour=v["false_alarms_per_hour"],
                coverage=v["coverage"],
            )
            for p, v in result["by_person"].items()
        ],
    )
    print(
        f"PRUEBA: {result['total']['detected']}/31 crisis, {100 * result['total']['sensitivity']:.2f}%, {result['total']['false_alarms_per_hour']:.3f} FA/h",
        flush=True,
    )
    backup(drive)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "pilot", "train", "evaluate"), required=True)
    parser.add_argument("--drive", type=Path, required=True)
    parser.add_argument("--ram-gib", type=float, default=16)
    parser.add_argument("--reuse", type=Path)
    parser.add_argument("--dataset", type=Path, default=REPO / "dataset")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    package = pack_check()
    OUT.mkdir(exist_ok=True)
    causal.DATASET = args.dataset.resolve()
    config = read(REPO / "codigo/configuracion_adaptativa.json")
    records, split = inputs()
    lookup = {r["path"]: r for r in records}
    entries = build_index(records, split, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage in ("pilot", "train", "evaluate") and device.type == "cpu" and not args.allow_cpu:
        raise ValueError("Activa GPU en Colab. CPU solo con --allow-cpu explícito para comprobaciones.")
    with process_lock():
        restore(args.drive)
        protocol_sha = freeze(config, split, entries, package)
        if args.stage == "prepare":
            metadata = prepare_entries(entries, lookup, config)
            stats = training_stats(entries, metadata)
            atomic_json(OUT / "normalizacion.json", stats)
            atomic_json(OUT / "cache_manifest.json", metadata)
            atomic_json(OUT / "recortes.json", entries)
            rows = [
                dict(
                    path=e["path"],
                    person=e["person"],
                    role=e["role"],
                    start=s["start"],
                    stop=s["stop"],
                    kind=s["kind"],
                    offset_samples=s.get("offset_samples", 0),
                )
                for e in entries
                for s in e["segments"]
            ]
            write_rows(OUT / "recortes.csv", rows)
            bind_preparation(protocol_sha, create=True)
            atomic_json(
                OUT / "estado.json",
                dict(
                    stage="prepared",
                    records=len(entries),
                    training_events=113,
                    validation_events=24,
                    test_not_prepared=True,
                ),
            )
            backup(args.drive)
            return

        if not (OUT / "normalizacion.json").exists():
            raise ValueError("Ejecuta primero prepare.")
        metadata = read(OUT / "cache_manifest.json")
        validate_caches(entries, lookup, metadata)
        stats = read(OUT / "normalizacion.json")
        if stats != training_stats(entries, metadata):
            raise ValueError("Normalización no corresponde a entrenamiento.")
        protocol_sha = bind_preparation(protocol_sha)
        data = TrainingWindows(entries, lookup, None, config)

        if args.stage == "pilot":
            result = pilot(data, stats, config, device, protocol_sha)
            plot_training_signal(data, config)
            backup(args.drive)
            if not result["passed"]:
                raise ValueError(
                    "El piloto no redujo la pérdida al menos 20%. Ejecuta la última celda para conservar el diagnóstico; no se inició el entrenamiento largo."
                )
            return

        validation = [r for r in records if r["person"] in split["validation"]]
        if args.stage == "train":
            check = read(OUT / "piloto.json") if (OUT / "piloto.json").exists() else {}
            if not check.get("passed") or check.get("protocol_sha") != protocol_sha:
                raise ValueError(
                    "El piloto TRAIN no está aprobado para esta preparación. Conserva piloto.json y diagnostica antes de entrenar."
                )
            if (OUT / "decision_validacion.json").exists():
                print("La decisión ya está fijada. No se repetirá el entrenamiento.", flush=True)
                return
            points = {
                variant: train_variant(variant, data, validation, stats, config, device, args.ram_gib, args.drive, protocol_sha)
                for variant in config["variants"]
            }
            decision = decide(points, config)
            decision.update(
                protocol_sha=protocol_sha,
                all_selections_sha256={v: digest(OUT / (v + "_seleccion.json")) for v in config["variants"]},
            )
            if decision["selected"]:
                decision["selection_sha256"] = digest(OUT / (decision["selected"] + "_seleccion.json"))
            atomic_json(OUT / "decision_validacion.json", decision)
            figures(config, decision, lookup, stats)
            plot_training_signal(data, config)
            atomic_json(
                OUT / "estado.json",
                dict(stage="validation_complete", decision=decision["status"], selected=decision["selected"]),
            )
            backup(args.drive)
            print(json.dumps(decision, indent=2), flush=True)
            return

        decision = read(OUT / "decision_validacion.json")
        if decision["protocol_sha"] != protocol_sha:
            raise ValueError("Decisión de otro protocolo.")
        points = {}
        for variant in config["variants"]:
            if digest(OUT / (variant + "_seleccion.json")) != decision["all_selections_sha256"][variant]:
                raise ValueError("Selección alterada: " + variant)
            chosen = read(OUT / (variant + "_seleccion.json"))
            if (
                digest(OUT / (variant + "_mejor.pt")) != chosen["weights_sha256"]
                or digest(OUT / (variant + "_validacion.npz")) != chosen["predictions_sha256"]
            ):
                raise ValueError("Pesos o validación alterados: " + variant)
            predictions = load_predictions(OUT / (variant + "_validacion.npz"), validation, config)
            reproduced, _ = choose_alarm(validation, predictions, config)
            if (
                reproduced["total"] != chosen["total"]
                or reproduced["rule"] != chosen["rule"]
                or reproduced["threshold"] != chosen["threshold"]
            ):
                raise ValueError("Selección no reproducida desde predicciones.")
            points[variant] = reproduced
        if any(decide(points, config)[k] != decision[k] for k in ("status", "selected", "eligible")):
            raise ValueError("Decisión de validación no reproducida.")
        evaluate_test(records, split, lookup, config, stats, decision, args.drive, protocol_sha, device)
        figures(config, decision, lookup, stats)
        plot_training_signal(data, config)
        atomic_json(
            OUT / "estado.json",
            dict(stage="closed", decision=decision["status"], test_performed=decision["status"] == "accepted"),
        )
        backup(args.drive)


if __name__ == "__main__":
    main()
