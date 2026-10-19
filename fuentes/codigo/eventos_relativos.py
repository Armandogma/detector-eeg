"""Conservative one-to-one overlap scoring and unchanged onset-only audit.

Not a complete SzCORE implementation: zero temporal tolerance, no event merging
or artificial splitting. A long active alarm cannot detect several seizures.
"""
import numpy as np
from ajustes_modelo import causal_alarms


def match_intervals(events, emitted, overlap=True):
    remaining = set(range(len(events)))
    matches, delays = [], []
    for i, (start, stop) in enumerate(emitted):
        candidates = [j for j in sorted(remaining)
                      if (min(stop, events[j][1])-max(start, events[j][0]) > 0 if overlap
                          else events[j][0] <= start < events[j][1])]
        if candidates:
            j = candidates[0]; remaining.remove(j)
            matches.append([i, j]); delays.append(max(0, start-events[j][0]))
    return matches, delays


def event_row(record, prediction, threshold, rule):
    ends = prediction['ends']; first, last = float(ends[0]-1), float(ends[-1])
    events = [(a, b) for a, b in record['events'] if a >= first and b <= last]
    emitted = causal_alarms(ends, prediction['probabilities'], threshold, rule)
    matches, delays = match_intervals(events, emitted, True)
    onset_matches, onset_delays = match_intervals(events, emitted, False)
    ictal = sum(max(0, min(b, last)-max(a, first)) for a, b in record['events'])
    alarm_seconds = sum(b-a for a, b in emitted)
    ictal_alarm = sum(max(0, min(b, d)-max(a, c)) for a, b in emitted for c, d in record['events'])
    return dict(path=record['path'], person=record['person'], threshold=float(threshold), rule=rule,
                events=len(events), partial_events=len(record['events'])-len(events), detected=len(matches),
                false_alarms=len(emitted)-len(matches), alarms=len(emitted), nonictal_seconds=last-first-ictal,
                evaluated_seconds=last-first, recorded_seconds=record['duration'], alarm_seconds=alarm_seconds,
                excluded_known_events=0, delays=delays, alarm_intervals=emitted, matches=matches,
                coverage_interval=[first, last], nonictal_alarm_seconds=alarm_seconds-ictal_alarm,
                longest_alarm_seconds=max([b-a for a, b in emitted] or [0]),
                onset_only_detected=len(onset_matches), onset_only_matches=onset_matches,
                onset_only_delays=onset_delays,
                alarm_already_active_at_onset=sum(emitted[i][0] < events[j][0] for i, j in matches))


def strict_rows(rows):
    return [dict(row, detected=row['onset_only_detected'],
                 false_alarms=row['alarms']-row['onset_only_detected'],
                 matches=row['onset_only_matches'], delays=row['onset_only_delays']) for row in rows]


def sample_metrics(records, predictions, threshold, rule):
    """Operational sample metrics, from exactly the alarm intervals already used."""
    from ajustes_modelo import causal_alarms
    tp = fn = fp = tn = 0
    for record, p in zip(records, predictions):
        active = np.zeros(len(p['ends']), dtype=bool)
        for a, b in causal_alarms(p['ends'], p['probabilities'], threshold, rule):
            active |= (p['ends']-1 < b) & (p['ends'] > a)
        y = p['labels'].astype(bool)
        tp += int(np.sum(active & y)); fn += int(np.sum(~active & y))
        fp += int(np.sum(active & ~y)); tn += int(np.sum(~active & ~y))
    return dict(tp=tp, fn=fn, fp=fp, tn=tn,
                sensitivity=tp/max(1, tp+fn), specificity=tn/max(1, tn+fp),
                precision=tp/max(1, tp+fp), f1=2*tp/max(1, 2*tp+fp+fn))


def ranking_metrics(records, predictions):
    from sklearn.metrics import average_precision_score, roc_auc_score
    result={}
    for person in sorted({r['person'] for r in records}):
        subset=[p for r,p in zip(records,predictions) if r['person']==person]
        labels=np.concatenate([p['labels'] for p in subset])
        scores=np.concatenate([p['probabilities'] for p in subset])
        result[person]=dict(AP=float(average_precision_score(labels,scores)),
                            AUROC=float(roc_auc_score(labels,scores)),
                            seizure_seconds=int(labels.sum()),evaluated_seconds=int(len(labels)))
    return result
