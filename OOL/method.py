"""Out-of-Limit (OOL) anomaly detection method.

The OOL method is the simplest threshold-based approach: a fixed upper
and lower limit is derived from a baseline window (typically mean ± k
standard deviations) and every value outside those limits is flagged
as an anomaly.  It does not adapt over time, which is exactly the weak
point the EWMA and MD methods are meant to address.

This module mirrors the structure of EWMA/method.py and MD/method.py so
that all three can be loaded and compared by the evaluation framework.
"""
import os
import time
import numpy as np
from cleaning_utils import extract_all_features
from db_utils import fetch_and_append

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_BASE_DIR, '..', 'Test_Data', 'raw_data.csv')

THRESHOLD = 3.0
POLL_INTERVAL = 60
WARMUP = 10

DB_CONN_PARAMS = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '5432'),
    'dbname': os.environ.get('DB_NAME', 'metrics'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', ''),
}

USE_DB = True


######
# OOL state: {n, mean, m2, frozen_mean, frozen_std}
######

def ool_init(threshold=THRESHOLD, warmup=WARMUP):
    return {'threshold': threshold, 'warmup': warmup,
            'n': 0, 'mean': 0.0, 'm2': 0.0,
            'frozen_mean': None, 'frozen_std': None}


######
# Feed one value, return (state, is_anomaly, distance). Limits freeze at warmup end.
######

def ool_update(state, value):
    value = float(value)
    threshold = state['threshold']
    warmup = state['warmup']

    state['n'] += 1
    delta = value - state['mean']
    state['mean'] += delta / state['n']
    delta2 = value - state['mean']
    state['m2'] += delta * delta2

    if state['n'] == warmup:
        state['frozen_mean'] = state['mean']
        state['frozen_std'] = np.sqrt(state['m2'] / (state['n'] - 1)) if state['n'] > 1 else 0.0

    if state['n'] <= warmup or state['frozen_std'] is None or state['frozen_std'] == 0:
        return state, False, 0.0

    distance = abs(value - state['frozen_mean']) / state['frozen_std']
    is_anomaly = distance > threshold
    return state, is_anomaly, distance


######
# Batch mode: run OOL on a full dataset, return list of anomaly dicts
######

def run_batch(features, timestamps, threshold=THRESHOLD, warmup=WARMUP):
    anomalies = []
    for name, values in features.items():
        state = ool_init(threshold, warmup)
        for i, value in enumerate(values):
            state, is_anomaly, distance = ool_update(state, value)
            if is_anomaly:
                anomalies.append({
                    'timestamp': timestamps[i],
                    'feature': name,
                    'value': float(value),
                    'mean': state['frozen_mean'],
                    'std': state['frozen_std'],
                    'distance': distance,
                })
    return anomalies


######
# Live monitor: poll CSV every minute, flag anomalies on new rows
######

def run_monitor(data_path=DATA_PATH, threshold=THRESHOLD,
                poll_interval=POLL_INTERVAL, use_db=USE_DB,
                conn_params=DB_CONN_PARAMS):
    features, _ = extract_all_features(data_path)
    states = {name: ool_init(threshold) for name in features.keys()}
    seen = set()

    print(f"Starting OOL monitor on {data_path}")
    print(f"threshold={threshold}, poll={poll_interval}s")
    if use_db:
        print(f"DB fetch enabled: {conn_params['host']}:{conn_params['port']}/{conn_params['dbname']}")
    print("Waiting for new minute data...\n")

    while True:
        if use_db:
            try:
                n = fetch_and_append(conn_params, data_path)
                if n:
                    print(f"  fetched {n} row(s) from DB {time.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception as e:
                print(f"  DB fetch failed: {e}")

        current_features, current_timestamps = extract_all_features(data_path)

        for i, ts in enumerate(current_timestamps):
            if ts in seen:
                continue
            seen.add(ts)
            for name, values in current_features.items():
                value = values[i]
                states[name], is_anomaly, distance = ool_update(states[name], value)
                if is_anomaly:
                    fm = states[name]['frozen_mean']
                    fs = states[name]['frozen_std']
                    print(f"[{ts}] ANOMALY  {name}: value={value:.4f} "
                          f"limit={fm:.4f} ± {threshold * fs:.4f} "
                          f"distance={distance:.2f}")
                    with open("anomalies.csv", "a", encoding="utf-8") as f:
                        f.write(f"{ts},{name}: value={value:.4f}, "
                                f"limit={fm:.4f} ± {threshold * fs:.4f}, "
                                f"distance={distance:.2f}\n")

        time.sleep(poll_interval)


if __name__ == "__main__":
    run_monitor()
