"""Python mask — reproducible anomaly injection module.

Injects three scenarios into clean telemetry data so detection rate,
delay and false alarm rate can be measured:

  1. Sudden spike       — short sharp jump in one metric.
  2. Slow drift         — gradual upward trend that compounds over time.
  3. Correlation break  — two normally-correlated metrics driven apart.

Each injection returns a copy of the data with the anomaly applied,
plus a list of ground-truth interval dicts describing exactly when and
where the anomaly was injected.

Data layouts
------------
EWMA / OOL  :  dict {feature_name: [values...]}   +  [timestamps]
MD          :  list of vectors [[v1, v2, ...], ...]  +  [timestamps]

This module works on the dict layout and provides a helper to convert
the injected dict back into the list-of-vectors layout that MD expects.
"""
from __future__ import annotations

import copy
from typing import Any


######
# Standard deviation of a list (population std)
######

def _std(values):
    if not values:
        return 0.0
    m = sum(values) / len(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


######
# Inject a sudden spike into a single feature
######

def inject_spike(features, timestamps, feature, start_idx, duration=5,
                 magnitude=5.0, unit="std"):
    result = copy.deepcopy(features)
    ts = list(timestamps)
    intervals = []

    end_idx = min(start_idx + duration, len(ts))
    baseline = features[feature][:start_idx] if start_idx > 0 else features[feature]
    baseline_std = _std(baseline)

    for i in range(start_idx, end_idx):
        original = result[feature][i]
        if unit == "std":
            offset = magnitude * baseline_std
        elif unit == "mult":
            offset = original * magnitude
        else:
            offset = magnitude
        result[feature][i] = original + offset

    intervals.append({
        'start_idx': start_idx,
        'end_idx': end_idx,
        'scenario': 'spike',
        'feature': feature,
        'description': f"spike +{magnitude}{unit} for {duration} min",
    })
    return {'features': result, 'timestamps': ts, 'intervals': intervals}


######
# Inject a slow (linear) drift into a single feature, held after the ramp
######

def inject_drift(features, timestamps, feature, start_idx, duration=60,
                 total_increase=5.0, unit="std"):
    result = copy.deepcopy(features)
    ts = list(timestamps)
    intervals = []

    end_ramp = min(start_idx + duration, len(ts))
    baseline = features[feature][:start_idx] if start_idx > 0 else features[feature]
    baseline_std = _std(baseline)

    if unit == "std":
        total_offset = total_increase * baseline_std
    elif unit == "mult":
        total_offset = None
    else:
        total_offset = total_increase

    for i in range(start_idx, len(ts)):
        original = result[feature][i]
        if i < end_ramp:
            frac = (i - start_idx) / duration
        else:
            frac = 1.0
        if unit == "mult":
            offset = original * total_increase * frac
        else:
            offset = total_offset * frac
        result[feature][i] = original + offset

    intervals.append({
        'start_idx': start_idx,
        'end_idx': len(ts),
        'scenario': 'drift',
        'feature': feature,
        'description': f"drift +{total_increase}{unit} over {duration} min (held after)",
    })
    return {'features': result, 'timestamps': ts, 'intervals': intervals}


######
# Break correlation between two metrics: feature_a up, feature_b down
######

def inject_correlation_break(features, timestamps, feature_a, feature_b,
                             start_idx, duration=30, magnitude=3.0, unit="std"):
    result = copy.deepcopy(features)
    ts = list(timestamps)
    intervals = []

    end_idx = min(start_idx + duration, len(ts))
    baseline_a = features[feature_a][:start_idx] if start_idx > 0 else features[feature_a]
    baseline_b = features[feature_b][:start_idx] if start_idx > 0 else features[feature_b]
    std_a = _std(baseline_a)
    std_b = _std(baseline_b)

    for i in range(start_idx, end_idx):
        if unit == "std":
            off_a = magnitude * std_a
            off_b = magnitude * std_b
        elif unit == "mult":
            off_a = result[feature_a][i] * magnitude
            off_b = result[feature_b][i] * magnitude
        else:
            off_a = magnitude
            off_b = magnitude
        result[feature_a][i] = result[feature_a][i] + off_a
        result[feature_b][i] = result[feature_b][i] - off_b

    intervals.append({
        'start_idx': start_idx,
        'end_idx': end_idx,
        'scenario': 'correlation_break',
        'feature': 'multi',
        'description': f"correlation break {feature_a} up / {feature_b} down "
                       f"for {duration} min",
    })
    return {'features': result, 'timestamps': ts, 'intervals': intervals}


######
# Default injection config: scenarios at 25%, 50%, 75% of the data
######

def _default_injection_config(n):
    return {
        "spike": {
            "feature": "cpu_user_pct",
            "start_idx": int(n * 0.25),
            "duration": 5,
            "magnitude": 5.0,
            "unit": "std",
        },
        "drift": {
            "feature": "mem_util_pct",
            "start_idx": int(n * 0.50),
            "duration": 60,
            "total_increase": 5.0,
            "unit": "std",
        },
        "correlation_break": {
            "feature_a": "cpu_user_pct",
            "feature_b": "sys_load_avg_1",
            "start_idx": int(n * 0.75),
            "duration": 30,
            "magnitude": 3.0,
            "unit": "std",
        },
    }


######
# Apply all three injection scenarios to one dataset
######

def inject_all(features, timestamps, config=None):
    cfg = _default_injection_config(len(timestamps))
    if config:
        cfg.update(config)

    result = copy.deepcopy(features)
    all_intervals = []

    sp = inject_spike(result, timestamps, **cfg["spike"])
    result = sp['features']
    all_intervals.extend(sp['intervals'])

    dr = inject_drift(result, timestamps, **cfg["drift"])
    result = dr['features']
    all_intervals.extend(dr['intervals'])

    cb = inject_correlation_break(result, timestamps, **cfg["correlation_break"])
    result = cb['features']
    all_intervals.extend(cb['intervals'])

    return {'features': result, 'timestamps': list(timestamps),
            'intervals': all_intervals}


######
# Feature order must match MD/cleaning_utils.py extract_important_features
######

MD_FEATURE_ORDER = [
    "cpu_user_pct",
    "cpu_system_pct",
    "cpu_iowait_pct",
    "cpu_switches",
    "cpu_interrupts",
    "mem_util_pct",
    "mem_committed_as_kbytes",
    "sys_load_avg_1",
    "sys_proc_running",
    "sys_proc_count",
    "sys_swap_used_pct",
]


######
# Convert dict layout to list-of-vectors layout (for MD)
######

def dict_to_vector_layout(dict_features, timestamps):
    vectors = []
    for i in range(len(timestamps)):
        vectors.append([dict_features[f][i] for f in MD_FEATURE_ORDER])
    return vectors, list(timestamps)
