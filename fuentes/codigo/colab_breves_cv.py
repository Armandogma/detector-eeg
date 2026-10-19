"""Cinco rondas independientes de desarrollo y prueba externa por persona.

Revisión pilot_cudnn_train_mode_1: corrige TRAIN/eval del piloto para cuDNN.
Solo se ejecuta dentro del ZIP aislado. No entrena desde la raíz de la tesis.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.signal import butter, sosfilt
import torch
import colab_adaptativo as engine
from modelo_breves import BriefSeizureNet, TrainingSampler, extract_features, infer_record, training_loss
from eventos_relativos import sample_metrics, ranking_metrics, strict_rows
from rutas import REPO

FAMILY = 'tcn_gru_breves_cv_v1'
SOFTWARE_REVISION = 'pilot_cudnn_train_mode_1'
OUT = REPO/'entrenamiento'
CACHE = REPO/'preparados'
GIB = 1024**3
read, digest, atomic_json, save_torch = engine.read, engine.digest, engine.atomic_json, engine.save_torch


def pack_check():
    package = read(REPO/'paquete_manifest.json')
    if package['family'] != FAMILY or package.get('software_revision') != SOFTWARE_REVISION:
        raise ValueError('Usa el paquete Breves CV Corregido de esta entrega.')
    for name, expected in package['files'].items():
        if digest(REPO/name) != expected:
            raise ValueError('Fuente modificada: '+name)
    return package


def inputs():
    records, _ = engine.inputs()
    plan = read(REPO/'analisis/particiones_breves_cv.json')
    people = {r['person'] for r in records}
    tested, validated = [], []
    if len(plan['folds']) != 5:
        raise ValueError('Se requieren las cinco rondas prefijadas.')
    for fold in plan['folds']:
        groups = [set(fold[k]) for k in ('train', 'validation', 'test')]
        if [len(g) for g in groups] != [12, 4, 4] or set.union(*groups) != people:
            raise ValueError('Población o tamaño de grupos distinto.')
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError('Personas compartidas dentro de una ronda.')
        for role in ('train', 'validation', 'test'):
            subset = [r for r in records if r['person'] in fold[role]]
            if (len(subset) != fold['counts'][role]['edf'] or
                sum(len(r['events']) for r in subset) != fold['counts'][role]['events']):
                raise ValueError('Contabilidad distinta en '+role)
        tested.extend(fold['test']); validated.extend(fold['validation'])
    if len(tested) != 20 or set(tested) != people or len(validated) != 20 or set(validated) != people:
        raise ValueError('Cada persona debe probarse y validar una vez.')
    return records, plan


def configure():
    engine.cache_path = cache_path
    engine.CACHE = CACHE
    engine.OUT = OUT
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def cache_path(record):
    return CACHE/(record['path'].replace('/', '__')+'.breves.npy')


def full_entries(records):
    return [dict(path=r['path'], person=r['person'], role='shared_label_free',
        segments=[dict(start=0, stop=int(r['duration']), kind='continuous', offset_samples=0)]) for r in records]


def prepare_record(record, config, reader=None):
    reader = reader or engine.causal.read_calibrated
    CACHE.mkdir(parents=True, exist_ok=True)
    target = cache_path(record); meta_path = target.with_suffix('.json')
    contract = dict(source=record['path'], header=record['header_sha256'], duration=record['duration'],
        feature_module_sha256=digest(REPO/'codigo/modelo_breves.py'),
        inherited_module_sha256=digest(REPO/'codigo/modelo_adaptativo.py'),
        windows_seconds=[4, 2], features=96, dtype='float16', adaptive=config['adaptive'],
        filter={k: config[k] for k in ('fs', 'filter_design_order', 'filter_low_hz', 'filter_high_hz', 'filter_initial_state')})
    if target.exists() and meta_path.exists():
        meta = read(meta_path)
        if meta['contract'] != contract or digest(target) != meta['sha256']:
            raise ValueError('Caché incompatible o alterada: '+record['path'])
        return meta
    fs, duration = config['fs'], int(record['duration'])
    sos = butter(config['filter_design_order'], [config['filter_low_hz'], config['filter_high_hz']], btype='bandpass', fs=fs, output='sos')
    state = np.zeros((len(sos), 18, 2), dtype=np.float64)
    filtered = np.empty((18, duration*fs), dtype=np.float32)
    for start in range(0, duration, config['preprocessing_block_seconds']):
        stop = min(duration, start+config['preprocessing_block_seconds'])
        raw = reader(record, start, stop)
        if raw.shape != (18, (stop-start)*fs) or not np.isfinite(raw).all():
            raise ValueError('EEG inválido: '+record['path'])
        value, state = sosfilt(sos, raw, axis=-1, zi=state)
        filtered[:, start*fs:stop*fs] = value
    features, ends = extract_features(filtered, config)
    array = features.astype(np.float16)
    if not np.isfinite(array).all() or array.shape != (duration-3, 18, 96):
        raise ValueError('Características inválidas: '+record['path'])
    temp = target.with_name(target.stem+'.partial.npy')
    np.save(temp, array, allow_pickle=False); os.replace(temp, target)
    meta = dict(contract=contract, sha256=digest(target), windows=len(ends), shape=list(array.shape),
        labels_used=False, no_pooled_patient_fitting=True)
    atomic_json(meta_path, meta)
    return meta


def prepare(records, config, reader=None):
    missing_bytes = sum((int(r['duration'])-3)*18*96*2 for r in records if not cache_path(r).exists())
    if shutil.disk_usage(REPO).free < missing_bytes+2*GIB:
        raise ValueError('No hay espacio para las características pendientes y 2 GiB de margen.')
    if reader is None:
        for record in records:
            engine.check_edf(record)
    result = {}
    with ThreadPoolExecutor(max_workers=config['performance']['prepare_workers']) as pool:
        for i, (record, meta) in enumerate(zip(records, pool.map(lambda r: prepare_record(r, config, reader), records)), 1):
            result[record['path']] = meta
            if i % 20 == 0 or i == len(records):
                print('Características 2 s + 4 s:', i, '/', len(records), 'EDF', flush=True)
    return result


def validate_caches(records, manifest):
    if set(manifest) != {r['path'] for r in records}:
        raise ValueError('Inventario de características incompleto.')
    for record in records:
        if digest(cache_path(record)) != manifest[record['path']]['sha256']:
            raise ValueError('Características distintas: '+record['path'])


def backup(destination):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in OUT.rglob('*') if p.is_file() and '.partial' not in p.name and p.suffix not in ('.lock', '.log'))
    hashes = {p.relative_to(OUT).as_posix(): digest(p) for p in files}
    temp = destination/'avance.partial.zip'
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_STORED) as archive:
        for p in files:
            archive.write(p, 'entrenamiento/'+p.relative_to(OUT).as_posix())
        archive.writestr('respaldo.json', json.dumps(dict(family=FAMILY, files=hashes)))
    with zipfile.ZipFile(temp) as archive:
        if archive.testzip():
            raise ValueError('Respaldo incompleto.')
    if (destination/'avance.zip').exists():
        os.replace(destination/'avance.zip', destination/'avance_anterior.zip')
    os.replace(temp, destination/'avance.zip')


def restore(destination):
    found = False
    for filename in ('avance.zip', 'avance_anterior.zip'):
        path = Path(destination)/filename
        if not path.exists():
            continue
        found = True
        try:
            with zipfile.ZipFile(path) as archive:
                if archive.testzip():
                    raise ValueError('CRC incorrecto.')
                meta = json.loads(archive.read('respaldo.json'))
                if meta['family'] != FAMILY or set(archive.namelist()) != {'respaldo.json'} | {'entrenamiento/'+n for n in meta['files']}:
                    raise ValueError('Familia/inventario de respaldo diferente.')
                payload = {}
                for name, wanted in meta['files'].items():
                    target = (OUT/name).resolve()
                    if '\\' in name or ':' in name or not target.is_relative_to(OUT.resolve()):
                        raise ValueError('Ruta de respaldo inválida.')
                    raw = archive.read('entrenamiento/'+name)
                    if hashlib.sha256(raw).hexdigest() != wanted:
                        raise ValueError('Respaldo alterado.')
                    payload[name] = raw
            for file in list(OUT.rglob('*')):
                if file.is_file() and file.suffix != '.lock' and file.relative_to(OUT).as_posix() not in payload:
                    target = REPO/'interrumpido'/file.relative_to(OUT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    while target.exists():
                        target = target.with_name(target.name+'.otra')
                    os.replace(file, target)
            for name, raw in payload.items():
                target = OUT/name; target.parent.mkdir(parents=True, exist_ok=True)
                temp = target.with_name(target.name+'.partial'); temp.write_bytes(raw); os.replace(temp, target)
            print('Recuperado:', filename, flush=True)
            return
        except (ValueError, KeyError, zipfile.BadZipFile) as error:
            print('No se recuperó', filename, str(error), flush=True)
    if found:
        raise ValueError('No hay un respaldo íntegro de esta familia. Conserva Drive.')


def freeze_root(config, plan, package):
    expected = dict(family=FAMILY, config=config, plan=plan, package=package,
        versions=engine.versions(), gpu=engine.resource_info()['gpu'], prior_test_exposure=True, clinical_validation=False)
    path = OUT/'protocolo.json'
    if path.exists():
        saved = read(path)
        if any(saved[k] != v for k, v in expected.items()):
            raise ValueError('Cambió el protocolo/entorno; no mezclar avances.')
    else:
        atomic_json(path, dict(**expected, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    return digest(path)


def bind_preparation(root_sha, metadata, create=False):
    path = OUT/'preparacion.json'
    expected = dict(protocol_sha256=root_sha, cache_manifest_sha256=digest(OUT/'cache_compartida.json'),
        label_free=True, feature_arrays_sha256={name: value['sha256'] for name, value in metadata.items()})
    if path.exists():
        if read(path) != expected:
            raise ValueError('Preparación congelada distinta.')
    elif create:
        atomic_json(path, expected)
    else:
        raise ValueError('Ejecuta primero prepare.')
    return digest(path)


def context(records, fold, config, prep_sha, create=False):
    folder = OUT/('grupo'+str(fold['fold'])); folder.mkdir(exist_ok=True)
    cfg = copy.deepcopy(config); cfg['seed'] += fold['fold']-1
    entries = [dict(path=r['path'], person=r['person'], role=role,
        segments=[dict(start=0, stop=int(r['duration']), offset_samples=0)])
        for r in records for role in ('train', 'validation') if r['person'] in fold[role]]
    payload = dict(fold=fold, config=cfg, preparation_sha256=prep_sha, entries=entries,
        normalization='Independent label-free past of each EDF; no fitting across patients or labels.')
    path = folder/'protocolo.json'
    if path.exists():
        if read(path) != payload:
            raise ValueError('Protocolo de ronda distinto.')
    elif create:
        atomic_json(path, payload)
    else:
        raise ValueError('Ronda no preparada.')
    return cfg, entries, digest(path), folder


def predict(model, records, config, device, cache=None):
    owned = cache is None
    cache = cache or engine.SignalCache(config['performance']['ram_cache_gib'], config['performance']['reserve_ram_gib'])
    result = []
    try:
        for i, record in enumerate(records, 1):
            ends, labels, _ = engine.causal.record_index(record, config)
            probabilities = infer_record(model, cache.get(record), config, device)
            if len(probabilities) != len(ends):
                raise ValueError('Reloj de predicciones distinto.')
            result.append(dict(ends=ends, labels=labels, probabilities=probabilities))
            if i % 30 == 0 or i == len(records):
                print('Inferencia:', i, '/', len(records), 'EDF', flush=True)
    finally:
        if owned:
            cache.close()
    return result


def selection_key(point, ap, config):
    return (int(engine.acceptable(point, config)), point['macro_sensitivity'], point['total']['sensitivity'],
            ap, -point['total']['false_alarms_per_hour'], -point['total']['time_under_alarm'])


def rng_snapshot(rng):
    return dict(numpy_generator=rng.bit_generator.state, torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(value, rng):
    rng.bit_generator.state = value['numpy_generator']
    torch.set_rng_state(value['torch_cpu'])
    if value['torch_cuda']:
        torch.cuda.set_rng_state_all(value['torch_cuda'])


def pilot(data, config, folder, protocol_sha, device, drive):
    path = folder/'piloto.json'
    if path.exists():
        previous = read(path)
        if previous['protocol_sha'] != protocol_sha or not previous['passed']:
            raise ValueError('Piloto de otro protocolo o rechazado; no repetirlo para ocultar un fallo.')
        return previous
    engine.deterministic(config['seed'])
    model = BriefSeizureNet(0, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['pilot']['learning_rate'])
    cache = engine.SignalCache(2, 2)
    rng = np.random.default_rng(config['seed'])
    try:
        x, labels, weights = (v.to(device) for v in data.batch(rng, cache, augment=False))
        model.eval()
        with torch.no_grad():
            initial = float(training_loss(model(x, return_heads=True), labels, weights))
        # cuDNN necesita un forward en modo TRAIN para calcular el backward de la GRU.
        model.train()
        losses = []
        for step in range(config['pilot']['steps']):
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(model(x, return_heads=True), labels, weights)
            if not torch.isfinite(loss):
                raise ValueError('Pérdida no finita en piloto.')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            final = float(training_loss(model(x, return_heads=True), labels, weights))
    finally:
        cache.close()
    value = dict(passed=final <= initial*config['pilot']['maximum_final_loss_ratio'], initial_loss=initial,
        final_loss=final, history=losses, protocol_sha=protocol_sha, weights_discarded=True, not_efficacy=True)
    atomic_json(path, value); backup(drive)
    if not value['passed']:
        raise ValueError('El piloto TRAIN falló. Se conserva el resultado y se detiene la corrida.')
    print('Piloto TRAIN aprobado; pesos descartados.', flush=True)
    return value


def train_fold(data, validation, config, folder, protocol_sha, device, drive):
    selected_path = folder/'seleccion.json'
    if selected_path.exists():
        return verify_selection(validation, config, folder, protocol_sha)
    pilot_path = folder/'piloto.json'
    if not pilot_path.exists() or read(pilot_path)['protocol_sha'] != protocol_sha or not read(pilot_path)['passed']:
        raise ValueError('Falta piloto aprobado en TRAIN de esta ronda.')
    engine.deterministic(config['seed'])
    model = BriefSeizureNet(config['training']['dropout'], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    rng = np.random.default_rng(config['seed'])
    checkpoint = folder/'ultimo.pt'; history = []; start = 0; best_key = None; best_epoch = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['protocol_sha'] != protocol_sha:
            raise ValueError('Checkpoint de otra ronda.')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        history, start = saved['history'], saved['epoch']
        best_key, best_epoch = tuple(saved['best_key']), saved['best_epoch']
        if digest(folder/'mejor.pt') != saved['best_weights_sha256']:
            raise ValueError('Pesos mejores distintos del checkpoint.')
        data.load_state_dict(saved['sampler_state']); restore_rng(saved['rng'], rng)
    cache = engine.SignalCache(config['performance']['ram_cache_gib'], config['performance']['reserve_ram_gib'])
    try:
        for epoch in range(start+1, config['training']['max_epochs']+1):
            model.train(); losses = []; tick = time.perf_counter()
            for step in range(1, config['training']['batches_per_epoch']+1):
                x, labels, weights = (v.to(device) for v in data.batch(rng, cache))
                optimizer.zero_grad(set_to_none=True)
                loss = training_loss(model(x, return_heads=True), labels, weights)
                if not torch.isfinite(loss):
                    raise ValueError('Pérdida no finita.')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip']); optimizer.step()
                losses.append(float(loss.detach()))
                if step % 50 == 0 or step == config['training']['batches_per_epoch']:
                    print(f"Época {epoch}/{config['training']['max_epochs']} · lote {step}/{config['training']['batches_per_epoch']} · pérdida {np.mean(losses):.4f}", flush=True)
            mined = data.mine(model, cache, device) if epoch in config['adaptive']['hard_mining_epochs'] else 0
            predictions = predict(model, validation, config, device, cache)
            ep_folder = folder/'epocas'; ep_folder.mkdir(exist_ok=True)
            ep_path = ep_folder/f'epoca{epoch:02d}_validacion.npz'
            engine.save_predictions(ep_path, validation, predictions)
            predictions = engine.load_predictions(ep_path, validation, config)
            ap = engine.macro_ap(validation, predictions)
            chosen, _ = engine.choose_alarm(validation, predictions, config)
            key = selection_key(chosen, ap, config)
            if best_key is None or key > best_key:
                best_key, best_epoch = key, epoch
                save_torch(folder/'mejor.pt', dict(model=model.state_dict(), epoch=epoch, protocol_sha=protocol_sha))
            compact = {k: v for k, v in chosen.items() if k != 'by_record'}
            history.append(dict(epoch=epoch, loss=float(np.mean(losses)), validation_macro_ap=ap,
                validation_point=compact, selection_key=list(key), validation_predictions_sha256=digest(ep_path),
                mined_hard_negatives=mined, seconds=time.perf_counter()-tick))
            payload = dict(protocol_sha=protocol_sha, epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                history=history, best_key=list(best_key), best_epoch=best_epoch, best_weights_sha256=digest(folder/'mejor.pt'),
                rng=rng_snapshot(rng), sampler_state=data.state_dict())
            save_torch(checkpoint, payload); atomic_json(folder/'historia.json', history)
            atomic_json(OUT/'estado_cv.json', dict(stage='training', group=folder.name, epoch=epoch,
                validation_detected=chosen['total']['detected'], validation_events=chosen['total']['events'],
                validation_fa_per_hour=chosen['total']['false_alarms_per_hour'], best_epoch=best_epoch, no_test_yet=True))
            backup(drive)
            print(f"VALIDACIÓN época {epoch}: {chosen['total']['detected']}/{chosen['total']['events']} · {chosen['total']['false_alarms_per_hour']:.3f} FA/h · mejor época {best_epoch}", flush=True)
    finally:
        cache.close()
    ep_path = folder/'epocas'/f'epoca{best_epoch:02d}_validacion.npz'
    shutil.copyfile(ep_path, folder/'validacion.npz')
    predictions = engine.load_predictions(folder/'validacion.npz', validation, config)
    chosen, curve = engine.choose_alarm(validation, predictions, config)
    ap = engine.macro_ap(validation, predictions)
    chosen.update(best_epoch=best_epoch, validation_macro_ap=ap, protocol_sha=protocol_sha,
        weights_sha256=digest(folder/'mejor.pt'), predictions_sha256=digest(folder/'validacion.npz'),
        history_sha256=digest(folder/'historia.json'), validation_goals_met=engine.acceptable(chosen, config))
    chosen['sample_metrics'] = sample_metrics(validation, predictions, chosen['threshold'], chosen['rule'])
    chosen['ranking_metrics_by_person'] = ranking_metrics(validation, predictions)
    atomic_json(folder/'seleccion.json', chosen); atomic_json(folder/'curva_validacion.json', curve)
    backup(drive)
    return chosen


def verify_selection(validation, config, folder, protocol_sha, deep=False):
    chosen = read(folder/'seleccion.json')
    if chosen['protocol_sha'] != protocol_sha or chosen['weights_sha256'] != digest(folder/'mejor.pt'):
        raise ValueError('Selección/pesos de otra ronda o alterados.')
    if chosen['predictions_sha256'] != digest(folder/'validacion.npz') or chosen['history_sha256'] != digest(folder/'historia.json'):
        raise ValueError('Historia/validación alteradas.')
    history = read(folder/'historia.json')
    if [h['epoch'] for h in history] != list(range(1, config['training']['max_epochs']+1)):
        raise ValueError('Faltan épocas completas.')
    for h in history:
        ep = folder/'epocas'/f"epoca{h['epoch']:02d}_validacion.npz"
        if digest(ep) != h['validation_predictions_sha256']:
            raise ValueError('Predicciones de época alteradas.')
        if list(selection_key(h['validation_point'], h['validation_macro_ap'], config)) != h['selection_key']:
            raise ValueError('Clave de selección de época distinta.')
        if deep:
            probabilities = engine.load_predictions(ep, validation, config)
            candidate, _ = engine.choose_alarm(validation, probabilities, config)
            if ({k: v for k, v in candidate.items() if k != 'by_record'} != h['validation_point'] or
                engine.macro_ap(validation, probabilities) != h['validation_macro_ap']):
                raise ValueError('Selección de época no reproducida.')
    best = max(history, key=lambda h: tuple(h['selection_key']))
    if best['epoch'] != chosen['best_epoch'] or best['validation_predictions_sha256'] != chosen['predictions_sha256']:
        raise ValueError('La época elegida no corresponde a la política.')
    predictions = engine.load_predictions(folder/'validacion.npz', validation, config)
    reproduced, curve = engine.choose_alarm(validation, predictions, config)
    for k, v in reproduced.items():
        if chosen[k] != v:
            raise ValueError('Alarma elegida no reproducida: '+k)
    if curve != read(folder/'curva_validacion.json'):
        raise ValueError('Curva de validación distinta.')
    if engine.macro_ap(validation, predictions) != chosen['validation_macro_ap']:
        raise ValueError('AP elegida distinta.')
    return chosen


def freeze_selections(contexts):
    selections = {}
    for fold, ctx in contexts.items():
        _, _, protocol_sha, folder = ctx
        chosen = read(folder/'seleccion.json')
        if chosen['protocol_sha'] != protocol_sha:
            raise ValueError('Selección de otra preparación.')
        selections[str(fold)] = dict(selection_sha256=digest(folder/'seleccion.json'), weights_sha256=digest(folder/'mejor.pt'),
            validation_sha256=digest(folder/'validacion.npz'), history_sha256=digest(folder/'historia.json'),
            validation_goals_met=chosen['validation_goals_met'])
    if len(selections) != 5:
        raise ValueError('Faltan modelos. No se abre prueba.')
    expected = dict(family=FAMILY, all_five_models_fixed_before_any_test=True, selections=selections,
        outer_policy='Evaluate every fold, including validation failures; no selective reporting.',
        prior_test_exposure=True, clinical_validation=False)
    path = OUT/'selecciones_congeladas.json'
    if path.exists() and read(path) != expected:
        raise ValueError('Selecciones cambiadas después de congelar.')
    atomic_json(path, expected)
    return expected


def verify_frozen(contexts, frozen):
    if (set(frozen['selections']) != {'1', '2', '3', '4', '5'} or
        not frozen.get('all_five_models_fixed_before_any_test')):
        raise ValueError('No hay cinco selecciones fijadas.')
    for fold, (_, _, protocol_sha, folder) in contexts.items():
        selected = read(folder/'seleccion.json'); saved = frozen['selections'][str(fold)]
        if selected['protocol_sha'] != protocol_sha:
            raise ValueError('Protocolo cambiado.')
        for filename, key in (('seleccion.json', 'selection_sha256'), ('mejor.pt', 'weights_sha256'),
                              ('validacion.npz', 'validation_sha256'), ('historia.json', 'history_sha256')):
            if digest(folder/filename) != saved[key]:
                raise ValueError('Archivos congelados modificados: '+filename)


def evaluate_fold(test, fold, ctx, frozen, device):
    config, _, protocol_sha, folder = ctx
    if (set(frozen['selections']) != {'1', '2', '3', '4', '5'} or
        not frozen.get('all_five_models_fixed_before_any_test') or
        read(OUT/'selecciones_congeladas.json') != frozen):
        raise ValueError('No se permite abrir una prueba antes de fijar los cinco modelos.')
    for number, fixed in frozen['selections'].items():
        group = OUT/('grupo'+number)
        for name, key in (('seleccion.json', 'selection_sha256'), ('mejor.pt', 'weights_sha256'),
                           ('validacion.npz', 'validation_sha256'), ('historia.json', 'history_sha256')):
            if digest(group/name) != fixed[key]:
                raise ValueError('Cambió un archivo de otra ronda antes de prueba.')
    saved = frozen['selections'][str(fold['fold'])]
    chosen = read(folder/'seleccion.json')
    if digest(folder/'seleccion.json') != saved['selection_sha256'] or digest(folder/'mejor.pt') != saved['weights_sha256']:
        raise ValueError('Selección o pesos alterados.')
    path, pred_path = folder/'evaluacion.json', folder/'prueba.npz'
    frozen_sha = digest(OUT/'selecciones_congeladas.json')
    if path.exists():
        previous = read(path)
        if previous['predictions_sha256'] != digest(pred_path) or previous['frozen_selections_sha256'] != frozen_sha:
            raise ValueError('Prueba previa alterada.')
    if not pred_path.exists():
        checkpoint = torch.load(folder/'mejor.pt', map_location='cpu', weights_only=True)
        if checkpoint['protocol_sha'] != protocol_sha or checkpoint['epoch'] != chosen['best_epoch']:
            raise ValueError('Pesos ajenos a la selección.')
        model = BriefSeizureNet(config['training']['dropout'], config).to(device)
        model.load_state_dict(checkpoint['model'])
        engine.save_predictions(pred_path, test, predict(model, test, config, device))
        del model
    predictions = engine.load_predictions(pred_path, test, config)
    result = engine.point(test, predictions, chosen['threshold'], chosen['rule'])
    if result['total']['events'] != fold['counts']['test']['events'] or result['total']['partial_events']:
        raise ValueError('Prueba incompleta.')
    result.update(fold=fold['fold'], predictions_sha256=digest(pred_path), protocol_sha=protocol_sha,
        selection_sha256=saved['selection_sha256'], weights_sha256=saved['weights_sha256'],
        frozen_selections_sha256=frozen_sha, validation_goals_met=saved['validation_goals_met'],
        prior_test_exposure=True, clinical_validation=False)
    result['sample_metrics'] = sample_metrics(test, predictions, chosen['threshold'], chosen['rule'])
    result['ranking_metrics_by_person'] = ranking_metrics(test, predictions)
    atomic_json(path, result)
    print(f"PRUEBA ronda {fold['fold']}: {result['total']['detected']}/{result['total']['events']} · {result['total']['false_alarms_per_hour']:.3f} FA/h", flush=True)
    return result


def summarize(results, records, artificial=False):
    rows = [row for result in results for row in result['by_record']]
    if len({r['path'] for r in rows}) != len(rows) or {r['path'] for r in rows} != {r['path'] for r in records}:
        raise ValueError('EDF omitidos o repetidos en prueba cruzada.')
    people = sorted({r['person'] for r in rows})
    total = engine.aggregate(rows)
    if not artificial and (len(rows) != 590 or len(people) != 20 or total['events'] != 168 or total['partial_events']):
        raise ValueError('La evaluación debe conservar toda la cohorte elegible.')
    by_person = {p: engine.aggregate([r for r in rows if r['person'] == p]) for p in people}
    onset = engine.aggregate(strict_rows(rows))
    lookup = {r['path']: r for r in records}; event_rows = []
    for row in rows:
        matches = {j: i for i, j in row['matches']}; strict = {j for _, j in row['onset_only_matches']}
        for j, (a, b) in enumerate(lookup[row['path']]['events']):
            alarm = row['alarm_intervals'][matches[j]] if j in matches else None
            event_rows.append(dict(person=row['person'], edf=row['path'], start=a, stop=b, duration=b-a,
                detected=j in matches, detected_onset=j in strict,
                delay=None if alarm is None else max(0, alarm[0]-a),
                already_active_at_onset=False if alarm is None else alarm[0] < a))
    durations = []
    for label, low, high in (('Hasta 10 s', 0, 10), ('Más de 10 a 30 s', 10, 30), ('Más de 30 s', 30, float('inf'))):
        selected = [r for r in event_rows if low < r['duration'] <= high]
        durations.append(dict(group=label, events=len(selected), detected=sum(r['detected'] for r in selected),
            sensitivity=None if not selected else sum(r['detected'] for r in selected)/len(selected)))
    sample_counts = {k: sum(r['sample_metrics'][k] for r in results) for k in ('tp', 'fn', 'fp', 'tn')}
    tp, fn, fp, tn = (sample_counts[k] for k in ('tp', 'fn', 'fp', 'tn'))
    sample_counts.update(sensitivity=tp/max(1, tp+fn), specificity=tn/max(1, tn+fp),
        precision=tp/max(1, tp+fp), f1=2*tp/max(1, 2*tp+fp+fn))
    folds = [dict(fold=r['fold'], total=r['total'], validation_goals_met=r['validation_goals_met']) for r in results]
    summary = dict(family=FAMILY, total=total, onset_only_total=onset, by_person=by_person, by_record=rows,
        onset_only_by_person={p: engine.aggregate(strict_rows([r for r in rows if r['person'] == p])) for p in people},
        nonictal_alarm_seconds=sum(r['nonictal_alarm_seconds'] for r in rows),
        longest_alarm_seconds=max(r['longest_alarm_seconds'] for r in rows),
        macro_sensitivity=float(np.mean([r['sensitivity'] for r in by_person.values()])),
        folds=folds, mean_fold_sensitivity=float(np.mean([r['total']['sensitivity'] for r in results])),
        std_fold_sensitivity=float(np.std([r['total']['sensitivity'] for r in results], ddof=1)),
        sample_metrics=sample_counts, ranking_metrics_by_person={p: v for r in results for p, v in r['ranking_metrics_by_person'].items()},
        event_duration_groups=durations, confidence_intervals=engine.bootstrap_people(rows, 1000, 2067),
        prior_test_exposure=True, clinical_validation=False, artificial=artificial,
        all_folds_reported_including_validation_failures=True,
        note='Each person tested once. No training sensitivity mixed into this denominator. Exploratory after prior test exposure.')
    atomic_json(OUT/'evaluacion_cv.json', summary)
    engine.write_rows(OUT/'eventos.csv', event_rows)
    engine.write_rows(OUT/'personas.csv', [dict(person=p, **{k: v[k] for k in ('events', 'detected', 'sensitivity', 'false_alarms_per_hour', 'event_precision', 'event_f1')}) for p, v in by_person.items()])
    engine.write_rows(OUT/'grupos.csv', [dict(fold=r['fold'], validation_goals_met=r['validation_goals_met'], **r['total']) for r in folds])
    figures(summary, artificial)
    return summary


def figures(summary, artificial=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
    prefix = 'DATOS ARTIFICIALES · ' if artificial else ''
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    validation = []
    for fold in range(1, 6):
        folder = OUT/('grupo'+str(fold))
        if (folder/'historia.json').exists():
            history = read(folder/'historia.json')
            axes[0].plot([h['epoch'] for h in history], [h['loss'] for h in history], marker='o', markersize=2, label=str(fold))
            axes[1].plot([h['epoch'] for h in history], [h['validation_macro_ap'] for h in history], marker='o', markersize=2, label=str(fold))
        if (folder/'seleccion.json').exists():
            validation.append((fold, read(folder/'seleccion.json')))
    axes[0].set(xlabel='Época completa', ylabel='Pérdida TRAIN')
    axes[1].set(xlabel='Época completa', ylabel='AP media VALIDACIÓN', ylim=(0, 1.02))
    for ax in axes:
        if ax.lines: ax.legend(title='Ronda')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if ax.lines and max(max(line.get_xdata()) for line in ax.lines) == 1:
            ax.set(xlim=(.5, 1.5), xticks=[1])
        ax.grid(alpha=.2)
    fig.suptitle(prefix+'Aprendizaje. AP no es sensibilidad por crisis'); fig.savefig(OUT/'aprendizaje.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, key, factor, label in ((axes[0], 'sensitivity', 100, 'Crisis detectadas (%)'), (axes[1], 'false_alarms_per_hour', 1, 'Falsas alarmas/h')):
        ax.bar([i for i, _ in validation], [factor*s['total'][key] for _, s in validation], color='#147D92')
        ax.set(xlabel='Ronda', ylabel=label)
    axes[1].axhline(1, color='#A55A32', linestyle='--')
    axes[1].set_ylim(0, max([1.15]+[s['total']['false_alarms_per_hour']*1.15 for _, s in validation]))
    fig.suptitle(prefix+'VALIDACIÓN · objetivos y fallos se conservan'); fig.savefig(OUT/'validacion.png', dpi=170); plt.close(fig)
    if summary is None:
        return
    people = list(summary['by_person'])
    fig, ax = plt.subplots(figsize=(12, 4.5), layout='constrained')
    bars = ax.bar(people, [100*summary['by_person'][p]['sensitivity'] for p in people], color='#147D92')
    for bar, person in zip(bars, people):
        value = summary['by_person'][person]
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f"{value['detected']}/{value['events']}", ha='center', fontsize=8)
    ax.set(ylim=(0, 114), ylabel='Crisis detectadas (%)', title=prefix+'PRUEBA CRUZADA · cada persona una sola vez')
    ax.tick_params(axis='x', rotation=60); fig.savefig(OUT/'personas.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, key, factor, label in ((axes[0], 'sensitivity', 100, 'Crisis detectadas (%)'), (axes[1], 'false_alarms_per_hour', 1, 'Falsas alarmas/h')):
        ax.bar([r['fold'] for r in summary['folds']], [factor*r['total'][key] for r in summary['folds']], color='#147D92')
        ax.set(xlabel='Ronda externa', ylabel=label)
    axes[1].set_ylim(0, max([1]+[r['total']['false_alarms_per_hour']*1.15 for r in summary['folds']]))
    for r in summary['folds']:
        value = r['total']['false_alarms_per_hour']
        axes[1].text(r['fold'], value+.02, f'{value:.2f}', ha='center', fontsize=9)
    fig.suptitle(prefix+'PRUEBA · se conservan las cinco rondas'); fig.savefig(OUT/'grupos.png', dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    delays = [d for r in summary['by_record'] for d in r['delays']]
    if delays: ax.hist(delays, bins=min(15, max(3, len(delays))), color='#777777', edgecolor='white')
    else: ax.text(.5, .5, 'Sin detecciones; demora no estimable', ha='center', transform=ax.transAxes)
    ax.set(xlabel='Demora desde el inicio anotado (s)', ylabel='Crisis detectadas', title=prefix+'PRUEBA · demora solo entre detectadas')
    fig.savefig(OUT/'demoras.png', dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    groups = summary['event_duration_groups']
    bars = ax.bar([g['group'] for g in groups], [100*(g['sensitivity'] or 0) for g in groups], color='#147D92')
    for bar, value in zip(bars, groups):
        text = f"{value['detected']}/{value['events']}" if value['events'] else 'Sin eventos'
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, text, ha='center')
    ax.set(ylim=(0, 114), ylabel='Crisis detectadas (%)', title=prefix+'PRUEBA · duración real de las crisis')
    fig.savefig(OUT/'duraciones.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    x = np.arange(len(people))
    primary = [summary['by_person'][p]['detected'] for p in people]
    strict = [summary['onset_only_by_person'][p]['detected'] for p in people]
    axes[0].bar(x-.18, primary, .36, label='Solapamiento uno a uno', color='#147D92')
    axes[0].bar(x+.18, strict, .36, label='Inicio dentro de crisis', color='#777777')
    axes[0].legend(fontsize=8, loc='upper left')
    axes[0].set(ylabel='Crisis detectadas', ylim=(0, max([1]+primary)*1.35), title='Mismas puntuaciones y regla')
    axes[1].bar(x, [100*summary['by_person'][p]['time_under_alarm'] for p in people], color='#A55A32')
    axes[1].set(ylabel='Tiempo bajo alarma (%)', title='Todo el tiempo marcado se conserva')
    for ax in axes:
        ax.set(xticks=x, xticklabels=people)
        ax.tick_params(axis='x', rotation=60, labelsize=8)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle(prefix+'PRUEBA · auditoría de alarmas. Duración máxima '+str(round(summary['longest_alarm_seconds'], 1))+' s')
    fig.savefig(OUT/'alarmas.png', dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('prepare', 'pilot', 'train', 'evaluate', 'verify'), required=True)
    parser.add_argument('--drive', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--ram-gib', type=float, default=24)
    args = parser.parse_args()
    package = pack_check(); configure(); records, plan = inputs()
    config = read(REPO/'codigo/configuracion_breves_cv.json')
    engine.causal.DATASET = args.dataset.resolve()
    OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.stage in ('pilot', 'train', 'evaluate') and device.type != 'cuda':
        raise ValueError('Activa GPU en Colab.')
    if args.ram_gib != config['performance']['ram_cache_gib']:
        raise ValueError('Mantén el límite de RAM fijado para este protocolo.')
    with engine.process_lock():
        restore(args.drive)
        root_sha = freeze_root(config, plan, package)
        if args.stage == 'prepare':
            metadata = prepare(records, config)
            path = OUT/'cache_compartida.json'
            if path.exists() and read(path) != metadata:
                raise ValueError('Caché compartida distinta del respaldo.')
            atomic_json(path, metadata)
            prep_sha = bind_preparation(root_sha, metadata, create=True)
            for fold in plan['folds']:
                context(records, fold, config, prep_sha, create=True)
            atomic_json(OUT/'estado_cv.json', dict(stage='prepared', people=20, edf=590, events=168,
                features_only_prepared=True, no_test_predictions_yet=True))
            backup(args.drive); return
        metadata = read(OUT/'cache_compartida.json'); validate_caches(records, metadata)
        prep_sha = bind_preparation(root_sha, metadata)
        contexts = {fold['fold']: context(records, fold, config, prep_sha) for fold in plan['folds']}
        lookup = {r['path']: r for r in records}
        if args.stage in ('pilot', 'train'):
            if (OUT/'selecciones_congeladas.json').exists():
                verify_frozen(contexts, read(OUT/'selecciones_congeladas.json'))
                print('Los cinco modelos ya están fijados; no se reentrenan.', flush=True); return
            for fold in plan['folds']:
                cfg, entries, protocol_sha, folder = contexts[fold['fold']]
                print('RONDA', fold['fold'], '/ 5 · solo TRAIN y VALIDACIÓN', flush=True)
                sampler = TrainingSampler(entries, lookup, cfg)
                pilot(sampler, cfg, folder, protocol_sha, device, args.drive)
                if args.stage == 'train':
                    validation = [r for r in records if r['person'] in fold['validation']]
                    train_fold(sampler, validation, cfg, folder, protocol_sha, device, args.drive)
                del sampler
            if args.stage == 'train':
                freeze_selections(contexts); figures(None); backup(args.drive)
            return
        frozen = read(OUT/'selecciones_congeladas.json'); verify_frozen(contexts, frozen)
        for fold in plan['folds']:
            cfg, _, protocol_sha, folder = contexts[fold['fold']]
            validation = [r for r in records if r['person'] in fold['validation']]
            verify_selection(validation, cfg, folder, protocol_sha, deep=args.stage == 'verify')
        if args.stage == 'verify':
            print('Todas las selecciones de época/alarma reproducidas solo desde VALIDACIÓN.', flush=True); return
        results = []
        for fold in plan['folds']:
            verify_frozen(contexts, frozen)
            test = [r for r in records if r['person'] in fold['test']]
            results.append(evaluate_fold(test, fold, contexts[fold['fold']], frozen, device))
            backup(args.drive)
        summary = summarize(results, records)
        atomic_json(OUT/'estado_cv.json', dict(stage='closed', folds=5, people=20, edf=590, events=168,
            all_folds_reported=True, all_models_fixed_before_test=True))
        backup(args.drive)
        t = summary['total']
        print(f"PRUEBA CRUZADA: {t['detected']}/168 · {100*t['sensitivity']:.2f}% · {t['false_alarms_per_hour']:.3f} FA/h", flush=True)


if __name__ == '__main__':
    main()
