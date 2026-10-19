"""Alarmas causales y emparejamiento uno a uno, con cobertura explícita."""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, auc


def precision_recall_area(labels, probabilities):
    if not np.any(labels):
        return None
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    return float(auc(recall, precision))


def alarms(ends, probabilities, threshold, open_steps=2, close_steps=2):
    result, active = [], None
    above = below = 0
    for time, probability in zip(ends, probabilities):
        if probability >= threshold:
            above += 1; below = 0
            if active is None and above >= open_steps:
                active = float(time)
        else:
            below += 1; above = 0
            if active is not None and below >= close_steps:
                result.append([active, float(time)]); active = None
    if active is not None:
        result.append([active, float(ends[-1])])
    return result


def event_metrics(record, ends, probabilities, threshold, config):
    if len(ends) == 0:
        raise ValueError('Sin ventanas evaluables')
    # Ventanas estiman el último segundo, cobertura [3, duración) para primer end=4.
    coverage_start, coverage_end = float(ends[0] - 1), float(ends[-1])
    evaluated = coverage_end - coverage_start
    events = [(a, b) for a, b in record['events'] if a >= coverage_start and b <= coverage_end]
    partial = [(a, b) for a, b in record['events'] if (a, b) not in events]
    emitted = alarms(ends, probabilities, threshold, config['alarm_open_steps'], config['alarm_close_steps'])
    remaining, delays, matched = set(range(len(events))), [], []
    for index, (start, stop) in enumerate(emitted):
        candidates = [i for i in sorted(remaining) if events[i][0] <= start < events[i][1]]
        if candidates:
            event = candidates[0]; remaining.remove(event)
            delays.append(start - events[event][0]); matched.append([index, event])
    # Crisis parcialmente cubiertas se reportan aparte; no son negativas para denominador.
    ictal = sum(max(0, min(b, coverage_end) - max(a, coverage_start)) for a, b in record['events'])
    return dict(path=record['path'], person=record['person'], threshold=float(threshold),
                events=len(events), partial_events=len(partial), detected=len(matched),
                false_alarms=len(emitted) - len(matched), alarms=len(emitted),
                nonictal_seconds=evaluated - ictal, evaluated_seconds=evaluated,
                recorded_seconds=record['duration'], alarm_seconds=sum(b-a for a, b in emitted),
                delays=delays, alarm_intervals=emitted, matches=matched,
                coverage_interval=[coverage_start, coverage_end])


def aggregate(rows):
    counts = {key: sum(r.get(key, 0) for r in rows) for key in (
        'events', 'partial_events', 'detected', 'false_alarms', 'alarms', 'nonictal_seconds',
        'evaluated_seconds', 'recorded_seconds', 'alarm_seconds', 'excluded_known_events',
        'excluded_records', 'unknown_annotation_records')}
    sensitivity = counts['detected'] / counts['events'] if counts['events'] else None
    precision = counts['detected'] / counts['alarms'] if counts['alarms'] else 0.0
    f1 = 2*precision*sensitivity/(precision+sensitivity) if sensitivity is not None and precision+sensitivity else 0.0
    delays = [d for r in rows for d in r['delays']]
    return dict(**counts, sensitivity=sensitivity, event_precision=precision, event_f1=f1,
                false_alarms_per_hour=counts['false_alarms']/(counts['nonictal_seconds']/3600)
                if counts['nonictal_seconds'] else None,
                coverage=counts['evaluated_seconds']/counts['recorded_seconds'],
                time_under_alarm=counts['alarm_seconds']/counts['evaluated_seconds'],
                delay_median=float(np.median(delays)) if delays else None,
                delay_q1=float(np.quantile(delays, .25)) if delays else None,
                delay_q3=float(np.quantile(delays, .75)) if delays else None)


def evaluate(records, predictions, threshold, config):
    if len(records) != len(predictions):
        raise ValueError('Número de registros/predicciones diferente')
    rows = [event_metrics(r, p['ends'], p['probabilities'], threshold, config)
            for r, p in zip(records, predictions)]
    all_labels = np.concatenate([p['labels'] for p in predictions])
    all_probs = np.concatenate([p['probabilities'] for p in predictions])
    total = aggregate(rows)
    total.update(segment_auc_pr=precision_recall_area(all_labels, all_probs),
                 segment_average_precision=float(average_precision_score(all_labels, all_probs))
                 if np.any(all_labels) else None,
                 segment_f1=float(f1_score(all_labels, all_probs >= threshold, zero_division=0)),
                 segment_prevalence=float(all_labels.mean()), segment_count=len(all_labels))
    return dict(total=total, by_record=rows,
                by_person={person: aggregate([r for r in rows if r['person'] == person])
                           for person in sorted({r['person'] for r in rows})})


def include_exclusions(result, records):
    """Cobertura del experimento formal: denominador incluye EDF excluidos."""
    rows = result['by_record'] + [dict(path=r['path'], person=r['person'], recorded_seconds=r['duration'],
                 events=0, partial_events=0, detected=0, false_alarms=0, alarms=0, nonictal_seconds=0,
                 evaluated_seconds=0, alarm_seconds=0, delays=[], excluded_records=1,
                 excluded_known_events=len(r['events']), unknown_annotation_records=int(r['annotation']=='desconocida'),
                 exclusion=r['exclusion']) for r in records]
    segment_fields = {k:v for k,v in result['total'].items() if k.startswith('segment_')}
    result['total'] = dict(**aggregate(rows), **segment_fields)
    result['by_record'] = rows
    result['by_person'] = {p: aggregate([r for r in rows if r['person']==p]) for p in sorted({r['person'] for r in rows})}
    return result


def choose_threshold(records, predictions, config):
    curve = []
    for threshold in config['threshold_candidates']:
        # Elegir por eventos no requiere ordenar millones de probabilidades
        # para recalcular AUC-PR y AP en cada umbral.
        rows = [event_metrics(r, p['ends'], p['probabilities'], threshold, config)
                for r, p in zip(records, predictions)]
        total = aggregate(rows)
        curve.append(dict(threshold=threshold, event_f1=total['event_f1'],
                          fa_per_hour=total['false_alarms_per_hour'], sensitivity=total['sensitivity']))
    # Desempate: menos falsas alarmas y después umbral más alto, sin observar prueba.
    best = max(curve, key=lambda r: (r['event_f1'], -(r['fa_per_hour'] or 0), r['threshold']))
    return best['threshold'], curve


def bootstrap_people(rows, repetitions=1000, seed=2026):
    people = sorted({r['person'] for r in rows})
    if len(people) < 2:
        return dict(status='No estimable con una sola persona; piloto técnico')
    rng = np.random.default_rng(seed)
    sampled = []
    for _ in range(repetitions):
        sample = rng.choice(people, len(people), replace=True)
        sampled.append(aggregate([r for person in sample for r in rows if r['person'] == person]))
    output = {}
    for key in ('sensitivity', 'false_alarms_per_hour', 'event_f1', 'delay_median', 'coverage'):
        values = [s[key] for s in sampled if s[key] is not None]
        output[key] = dict(low=float(np.quantile(values, .025)), high=float(np.quantile(values, .975)),
                           defined_resamples=len(values)) if values else None
    return dict(status='Remuestreo por persona', people=len(people), repetitions=repetitions, intervals=output)
