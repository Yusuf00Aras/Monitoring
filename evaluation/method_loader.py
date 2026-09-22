"""Method loader -- unified interface to run EWMA, MD, or OOL in batch mode.

MD now needs a baseline (baseline_features + baseline_timestamps) to fit
its frozen reference distribution (mean, inverse covariance) before it can
score anything -- that distribution is fit once and never updated, so the
caller must supply the clean baseline explicitly. EWMA and OOL don't need
this: they build their own state online (EWMA/OOL each warm up on the
start of whatever stream they're given).
"""
from __future__ import annotations

import importlib
import os
import sys

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
# Run a method in batch mode and return its raw anomaly list.
# MD requires baseline_features/baseline_timestamps (dict layout, same
# shape as dict_features) to fit its frozen reference distribution.
######

def run_method(method, dict_features, timestamps, threshold=None,
              baseline_features=None, baseline_timestamps=None, **kwargs):
    method = method.upper()
    mod = _import_method_module(method)

    if method == "EWMA":
        return mod.run_batch(dict_features, timestamps,
                             threshold=threshold if threshold is not None else 3.0,
                             **kwargs)
    elif method == "OOL":
        return mod.run_batch(dict_features, timestamps,
                             threshold=threshold if threshold is not None else 3.0,
                             **kwargs)
    elif method == "MD":
        from injection import dict_to_vector_layout
        if baseline_features is None or baseline_timestamps is None:
            raise ValueError(
                "MD requires baseline_features and baseline_timestamps to fit "
                "its frozen reference distribution (mean, inverse covariance)."
            )
        baseline_vectors, _ = dict_to_vector_layout(baseline_features, baseline_timestamps)
        vectors, ts = dict_to_vector_layout(dict_features, timestamps)
        return mod.run_batch(baseline_vectors, vectors, ts,
                             threshold=threshold if threshold is not None else 7.0,
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
