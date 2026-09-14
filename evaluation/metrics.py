"""Evaluation metrics — detection rate, detection delay, false alarms.

Compares the anomalies detected by a method against the ground-truth
injection intervals and computes:

  * Detection rate  — fraction of injected scenarios detected at least once.
  * Detection delay — minutes between injection start and first alert (mean).
  * False alarm rate — alerts outside any ground-truth interval, per hour.
"""
from __future__ import annotations


######
# Convert a method's raw anomaly dicts to a common {timestamp, index, feature, distance} format
######

def normalize_anomalies(raw, timestamps):
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}
    result = []
    for a in raw:
        ts = a["timestamp"]
        idx = ts_to_idx.get(ts)
        if idx is None:
            for t, i in ts_to_idx.items():
                if t[:19] == ts[:19]:
                    idx = i
                    break
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
# Score a method's detections against the ground-truth intervals
######

def evaluate(method_name, detected, intervals, total_minutes):
    scenario_results = []

    for iv in intervals:
        hits = [d for d in detected if iv['start_idx'] <= d['index'] < iv['end_idx']]
        if hits:
            first = min(hits, key=lambda d: d['index'])
            delay = first['index'] - iv['start_idx']
            scenario_results.append({
                'interval': iv,
                'detected': True,
                'delay_minutes': delay,
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

    false_alarms = 0
    for d in detected:
        inside = any(iv['start_idx'] <= d['index'] < iv['end_idx'] for iv in intervals)
        if not inside:
            false_alarms += 1

    hours = total_minutes / 60.0 if total_minutes > 0 else 1.0
    false_alarm_rate = false_alarms / hours

    return {
        'method': method_name,
        'total_injections': len(intervals),
        'detected_count': detected_count,
        'detection_rate': detection_rate,
        'mean_delay_minutes': mean_delay,
        'false_alarms': false_alarms,
        'false_alarm_rate_per_hour': false_alarm_rate,
        'total_minutes': total_minutes,
        'scenario_results': scenario_results,
        'all_detected': detected,
    }


######
# Normalise raw anomalies and run evaluate (main entry point)
######

def evaluate_raw(method_name, raw_anomalies, timestamps, intervals, total_minutes=None):
    detected = normalize_anomalies(raw_anomalies, timestamps)
    if total_minutes is None:
        total_minutes = len(timestamps)
    return evaluate(method_name, detected, intervals, total_minutes)


######
# Format an evaluation report dict as a human-readable summary string
######

def format_report(report):
    lines = [
        f"=== {report['method']} ===",
        f"  Detection rate : {report['detected_count']}/{report['total_injections']} "
        f"({report['detection_rate']:.1%})",
    ]
    if report['mean_delay_minutes'] is not None:
        lines.append(f"  Mean delay      : {report['mean_delay_minutes']:.1f} min")
    else:
        lines.append("  Mean delay      : N/A (nothing detected)")
    lines.append(
        f"  False alarms    : {report['false_alarms']} "
        f"({report['false_alarm_rate_per_hour']:.2f}/h over "
        f"{report['total_minutes'] / 60:.1f} h)"
    )
    for sr in report['scenario_results']:
        iv = sr['interval']
        status = "DETECTED" if sr['detected'] else "MISSED"
        delay = f"{sr['delay_minutes']} min" if sr['delay_minutes'] is not None else "—"
        lines.append(
            f"    [{iv['scenario']}] {status}  delay={delay}  "
            f"({iv['description']})"
        )
    return "\n".join(lines)
