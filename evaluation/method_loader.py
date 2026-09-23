"""Method loader -- unified interface to run EWMA, MD, or OOL in batch mode.

EWMA and MD need a baseline (baseline_features + baseline_timestamps) to
fit their frozen reference (EWMA: per-metric mean/std, MD: mean/inverse
covariance) before they can score anything -- the reference is fit once
and never updated, so the caller must supply the clean baseline
explicitly. OOL doesn't need this: its limits are static.
"""
from __future__ import annotations

import importlib
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
# Run a method in batch mode and return its raw anomaly list.
# EWMA and MD require baseline_features/baseline_timestamps (dict layout,
# same shape as dict_features) to fit their frozen reference.
######

def run_method(method, dict_features, timestamps, threshold=None,
              baseline_features=None, baseline_timestamps=None, **kwargs):
    method = method.upper()
    mod = _import_method_module(method)

    if method in ("EWMA", "MD") and (baseline_features is None
                                     or baseline_timestamps is None):
        raise ValueError(
            f"{method} requires baseline_features and baseline_timestamps to "
            "fit its frozen reference."
        )

    if method == "EWMA":
        return mod.run_batch(baseline_features, dict_features, timestamps,
                             threshold=threshold if threshold is not None else mod.THRESHOLD,
                             **kwargs)
    elif method == "OOL":
        # Static limits -- no threshold is passed or tuned.
        return mod.run_batch(dict_features, timestamps, **kwargs)
    elif method == "MD":
        from injection import dict_to_vector_layout
        baseline_vectors, _ = dict_to_vector_layout(baseline_features, baseline_timestamps)
        vectors, ts = dict_to_vector_layout(dict_features, timestamps)
        return mod.run_batch(baseline_vectors, vectors, ts,
                             threshold=threshold if threshold is not None else mod.THRESHOLD,
                             **kwargs)
    else:
        raise ValueError(f"Unknown method: {method!r}. Expected 'EWMA', 'MD', or 'OOL'.")


######
# Return the data layout a method expects: 'dict' or 'vector'
######

def get_method_layout(method):
    method = method.upper()
    if method in ("EWMA", "OOL"):
        return "dict"
    elif method == "MD":
        return "vector"
    raise ValueError(f"Unknown method: {method!r}")


######
# Return the list of available methods
######

def list_methods():
    return ["EWMA", "MD", "OOL"]
