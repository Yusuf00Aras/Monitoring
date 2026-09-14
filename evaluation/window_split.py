"""Window split — divide data into baseline and evaluation windows.

The baseline window is the first baseline_ratio of the data (default 50%)
and the evaluation window is the remainder. Works for both layouts:
  * dict  {feature: [values]}   (EWMA / OOL)
  * list  [[v1, v2, ...], ...]  (MD)
"""
from __future__ import annotations


######
# Split data into baseline and evaluation windows, return a dict
######

def split_windows(features, timestamps, baseline_ratio=0.5):
    n = len(timestamps)
    split = int(n * baseline_ratio)

    if isinstance(features, dict):
        baseline_features = {k: v[:split] for k, v in features.items()}
        eval_features = {k: v[split:] for k, v in features.items()}
    else:
        baseline_features = features[:split]
        eval_features = features[split:]

    return {
        'baseline_features': baseline_features,
        'baseline_timestamps': timestamps[:split],
        'eval_features': eval_features,
        'eval_timestamps': timestamps[split:],
        'split_index': split,
    }


######
# Concatenate baseline and evaluation features back into one dataset
######

def merge_windows(baseline_features, eval_features, layout="dict"):
    if layout == "dict":
        merged = {}
        for k in baseline_features:
            merged[k] = list(baseline_features[k]) + list(eval_features[k])
        return merged
    else:
        return list(baseline_features) + list(eval_features)
