"""Evaluation metrics -- detection rate, detection delay, false alarms.

Compares the anomalies detected by a method against the ground-truth
injection intervals.

Rules
-----
* An alarm only counts as a detection if it lies inside the injection
  interval AND concerns an injected feature (MD alarms are 'multi' and
  always match). An EWMA/OOL alarm on an unrelated metric is a false alarm.
* Only alarms inside the evaluation window (index >= eval_start) are
  counted. The baseline window is used for calibration only.
* Alarms on an injected feature within `grace` minutes after the interval
  ends (e.g. EWMA reacting to the value dropping back) are ignored --
  they are caused by the injection but are not a detection.
"""
from __future__ import annotations


######
# Convert raw anomaly dicts to {timestamp, index, feature, distance}
######

def normalize_anomalies(raw, timestamps):
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}
    prefix_to_idx = {ts[:19]: i for i, ts in enumerate(timestamps)}
    result = []
    for a in raw:
        ts = a["timestamp"]
        idx = ts_to_idx.get(ts, prefix_to_idx.get(ts[:19]))
        if idx is None:
            continue
        result.append({
            'timestamp': ts,
            'index': idx,
            'feature': a.get("feature", "multi"),
            'distance': a.get("distance", 0.0),
        })
    return result


######
# Does an alarm concern the injected feature(s) of an interval?
######

def _matches_feature(d, iv):
    feat = iv.get('feature')
    feats = iv.get('features')
    if feats is None and feat is not None and feat != 'multi':
        feats = [feat]
    return d['feature'] == 'multi' or not feats or d['feature'] in feats


def _in_interval(d, iv, grace=0):
    return iv['start_idx'] <= d['index'] < iv['end_idx'] + grace and _matches_feature(d, iv)


######
# Score a method's detections against the ground-truth intervals
######

def evaluate(method_name, detected, intervals, total_minutes, eval_start=0, grace=5):
    detected = [d for d in detected if d['index'] >= eval_start]
    scenario_results = []

    for iv in intervals:
        hits = [d for d in detected if _in_interval(d, iv)]
        if hits:
            first = min(hits, key=lambda d: d['index'])
            scenario_results.append({
                'interval': iv,
                'detected': True,
                'delay_minutes': first['index'] - iv['start_idx'],
                'first_detection_index': first['index'],
            })
        else:
            scenario_results.append({
                'interval': iv,
                'detected': False,
                'delay_minutes': None,
                'first_detection_index': None,
            })

    detected_count = sum(1 for sr in scenario_results if sr['detected'])
    detection_rate = detected_count / len(intervals) if intervals else 0.0

    delays = [sr['delay_minutes'] for sr in scenario_results if sr['detected']]
    mean_delay = sum(delays) / len(delays) if delays else None

    false_alarms = sum(
        1 for d in detected
        if not any(_in_interval(d, iv, grace) for iv in intervals)
    )

    hours = total_minutes / 60.0 if total_minutes > 0 else 1.0

    return {
        'method': method_name,
        'total_injections': len(intervals),
        'detected_count': detected_count,
        'detection_rate': detection_rate,
        'mean_delay_minutes': mean_delay,
        'false_alarms': false_alarms,
        'false_alarm_rate_per_hour': false_alarms / hours,
        'total_minutes': total_minutes,
        'scenario_results': scenario_results,
        'all_detected': detected,
    }


######
# Normalise raw anomalies and run evaluate (main entry point)
######

def evaluate_raw(method_name, raw_anomalies, timestamps, intervals,
                 total_minutes=None, eval_start=0, grace=5):
    detected = normalize_anomalies(raw_anomalies, timestamps)
    if total_minutes is None:
        total_minutes = len(timestamps) - eval_start
    return evaluate(method_name, detected, intervals, total_minutes, eval_start, grace)
