"""Method loader — unified interface to run EWMA, MD, or OOL in batch mode.

Loads the correct method module, converts the data to the layout the
method expects, calls run_batch() with the right arguments, and returns
the raw anomaly list in a common format.

Layouts
-------
EWMA / OOL  :  dict {feature_name: [values...]}
MD          :  list of vectors [[v1, v2, ...], ...]

The evaluation framework always works in the dict layout; this loader
handles the conversion for MD.
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
# Run a method in batch mode and return its raw anomaly list
######

def run_method(method, dict_features, timestamps, threshold=None, **kwargs):
    method = method.upper()
    mod = _import_method_module(method)

    if method == "EWMA":
        return mod.run_batch(dict_features, timestamps,
                             threshold=threshold if threshold is not None else 3.0,
                             **kwargs)
    elif method == "OOL":
        # OOL uses static limits — threshold is not used, but accepted for interface compatibility
        return mod.run_batch(dict_features, timestamps, **kwargs)
    elif method == "MD":
        from injection import dict_to_vector_layout
        vectors, ts = dict_to_vector_layout(dict_features, timestamps)
        return mod.run_batch(vectors, ts,
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
