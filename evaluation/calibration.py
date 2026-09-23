"""Calibration -- tune each method's threshold to a target false alarm rate.

Each method has a single threshold parameter. By running each method on
a clean baseline window (no injections) and counting the false alarms,
we binary-search for the threshold that produces the desired false alarm
rate. All three methods are then compared at that same rate.

For EWMA and MD, the same baseline window is used both to fit the frozen
reference and to score it (count false alarms) during calibration --
there are no injected anomalies in the baseline, so fitting and scoring
on the same clean data is consistent with how both are fit once for real
evaluation runs too. Alarms are counted per excursion for all methods.
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Callable

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


######
# Import a method's module, isolating cleaning_utils/db_utils name collisions
######

def _import_method_module(method):
    method = method.upper()

    for m in ("EWMA", "MD", "OOL"):
        d = os.path.join(_PROJECT_ROOT, m)
        while d in sys.path:
            sys.path.remove(d)

    for mod_name in ("cleaning_utils", "db_utils"):
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    for m in ("EWMA", "MD", "OOL"):
        for key in (f"{m}.method", m):
            if key in sys.modules:
                del sys.modules[key]

    module_dir = os.path.join(_PROJECT_ROOT, method)
    sys.path.insert(0, module_dir)
    return importlib.import_module(f"{method}.method")


######
# Binary search for the threshold producing target_false_alarms on clean data
######

def calibrate_threshold(run_fn, features, timestamps, target_false_alarms=0,
                        threshold_low=0.5, threshold_high=20.0,
                        tolerance=0.01, max_iter=50):
    lo, hi = threshold_low, threshold_high
    best = hi
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        anomalies = run_fn(threshold=mid, features=features, timestamps=timestamps)
        n_alarms = len(anomalies)

        if n_alarms <= target_false_alarms:
            best = mid
            hi = mid
        else:
            lo = mid

        if hi - lo < tolerance:
            break

    return best


######
# Calibrate the EWMA threshold on a clean baseline window
######

def calibrate_ewma(features, timestamps, target_false_alarms=0,
                   alpha=None, **kwargs):
    # Same procedure the live EWMA monitor uses (EWMA/method.py).
    mod = _import_method_module("EWMA")
    if alpha is None:
        alpha = mod.ALPHA
    return mod.calibrate_threshold(features, timestamps, target_false_alarms,
                                   alpha=alpha, **kwargs)


######
# Calibrate the MD threshold on a clean baseline window (vector layout).
# The baseline is used both to fit MD's frozen reference distribution and
# to score itself for false-alarm counting.
######

def calibrate_md(features, timestamps, target_false_alarms=0, **kwargs):
    # Same procedure the live MD monitor uses (MD/method.py).
    mod = _import_method_module("MD")
    return mod.calibrate_threshold(features, timestamps, target_false_alarms, **kwargs)


######
# Calibrate EWMA and MD to the same false alarm rate.
# OOL is not calibrated -- its limits are static.
######

def calibrate_all(baseline_dict_features, baseline_vector_features,
                  timestamps, target_false_alarms=0):
    print("Calibrating thresholds to target false alarms =", target_false_alarms)
    ewma_thr = calibrate_ewma(baseline_dict_features, timestamps, target_false_alarms)
    print(f"  EWMA threshold = {ewma_thr:.4f}")

    md_thr = calibrate_md(baseline_vector_features, timestamps, target_false_alarms)
    print(f"  MD   threshold = {md_thr:.4f}")

    return {"EWMA": ewma_thr, "MD": md_thr}
