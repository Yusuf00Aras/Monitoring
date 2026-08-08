import csv, json
import numpy as np

PREFIX = {"zbx_cpu": "cpu", "zbx_filesystem": "fs", "zbx_memory": "mem", "zbx_system": "sys"}

def load_test_data(path, max_col_nan=0.05):
    rows = {}  # timestamp -> {feature: wert}
    with open(path, newline="") as f:
        for r in csv.reader(f):
            if len(r) < 4:
                continue
            ts, mtype, tags_raw, metrics_raw = r[0], r[1], r[2], r[3]
            try:
                metrics = json.loads(metrics_raw)
            except (ValueError, TypeError):
                continue
            prefix = PREFIX.get(mtype, mtype)
            tags = json.loads(tags_raw) if tags_raw.strip() else {}
            if "path" in tags:
                p = "root" if tags["path"] == "/" else tags["path"].strip("/").replace("/", "_")
                prefix = f"{prefix}_{p}"
            for k, v in metrics.items():
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    continue  # Text-/Bool-Metriken überspringen
                rows.setdefault(ts, {})[f"{prefix}_{k}".replace(".", "_")] = v

    timestamps = sorted(rows)
    features = sorted({k for row in rows.values() for k in row})
    data = np.array([[rows[t].get(f, np.nan) for f in features] for t in timestamps], dtype=float)

    # --- Bereinigung: verhindert NaN/inf in np.cov (sonst "SVD did not converge") ---
    data[~np.isfinite(data)] = np.nan
    col_ok = np.isnan(data).mean(axis=0) <= max_col_nan   # chronisch fehlende Spalten raus
    dropped_cols = [f for f, k in zip(features, col_ok) if not k]
    data = data[:, col_ok]
    features = [f for f, k in zip(features, col_ok) if k]
    row_ok = ~np.isnan(data).any(axis=1)                  # restliche unvollständige Minuten raus
    dropped_rows = int((~row_ok).sum())
    data = data[row_ok]
    timestamps = [t for t, k in zip(timestamps, row_ok) if k]

    print(f"[load_test_data] {data.shape[0]} Minuten x {data.shape[1]} Features "
          f"| Spalten verworfen: {dropped_cols} | Minuten verworfen: {dropped_rows}")
    return data, timestamps, features