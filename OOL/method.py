"""Out-of-Limit (OOL) anomaly detection method.

Limits are no longer a fixed static value (e.g. "90%"). Instead, each
feature's limit is derived from its own data: during a warmup period the
mean and standard deviation are estimated online (Welford's algorithm),
then frozen. From that point on, a value is out-of-limit if it falls
more than `threshold` standard deviations above or below the frozen
mean. An anomaly is only flagged once a value stays out-of-limit for
`sustained_minutes` consecutive minutes (same sustained-duration idea
as before, kept because it matches how the Introduction describes the
production monitoring's behavior).

Because the limit is now derived statistically rather than hand-picked
per metric type, OOL can run on ALL features (not just the five
percentage-based ones that had a sensible fixed "90%" ceiling before).
"""
import os
import time
import logging
import numpy as np
from cleaning_utils import extract_all_features
from db_utils import fetch_and_append

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_BASE_DIR, '..', 'Test_Data', 'raw_data.csv')
LOG_PATH = os.path.join(_BASE_DIR, 'monitor.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)

# Number of standard deviations away from the frozen baseline mean
# before a value counts as "out of limit". Calibrated on baseline data
# by evaluation/calibration.py, same as EWMA's and MD's thresholds.
THRESHOLD = 3.0

# Minutes used to estimate the frozen baseline mean/std per feature.
WARMUP = 60

# An anomaly is flagged after the value stays out-of-limit for this
# many consecutive minutes.
SUSTAINED_MINUTES = 5

POLL_INTERVAL = 60

DB_CONN_PARAMS = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '5432'),
    'dbname': os.environ.get('DB_NAME', 'metrics'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', ''),
}

USE_DB = True


######
# OOL state: {threshold, warmup, sustained_minutes, n, frozen_mean,
#             frozen_std, consecutive, alerted, running mean/M2}
######

def ool_init(threshold=THRESHOLD, warmup=WARMUP, sustained_minutes=SUSTAINED_MINUTES):
    return {
        'threshold': threshold,
        'warmup': warmup,
        'sustained_minutes': sustained_minutes,
        'n': 0,
        'frozen_mean': None,
        'frozen_std': None,
        'consecutive': 0,
        'alerted': False,
        # Welford accumulators used only during warmup, then frozen.
        '_running_mean': 0.0,
        '_running_m2': 0.0,
    }


######
# Feed one value, return (state, is_anomaly, distance).
#
# During the first `warmup` minutes: accumulate mean/std online, flag
# nothing. After warmup: mean/std are frozen (no longer updated), and a
# value is "out of limit" if it is more than `threshold` standard
# deviations from the frozen mean. An anomaly fires once that has held
# for `sustained_minutes` consecutive minutes.
######

def ool_update(state, value):
    value = float(value)
    state['n'] += 1

    if state['n'] <= state['warmup']:
        # Welford's online mean/variance, restricted to the warmup window.
        delta = value - state['_running_mean']
        state['_running_mean'] += delta / state['n']
        delta2 = value - state['_running_mean']
        state['_running_m2'] += delta * delta2

        if state['n'] == state['warmup']:
            variance = state['_running_m2'] / state['n'] if state['n'] > 1 else 0.0
            state['frozen_mean'] = state['_running_mean']
            state['frozen_std'] = float(np.sqrt(variance))

        return state, False, 0.0

    mean = state['frozen_mean']
    std = state['frozen_std']

    if std > 0:
        upper = mean + state['threshold'] * std
        lower = mean - state['threshold'] * std
        outside = value > upper or value < lower
        distance = (value - mean) / std
    else:
        # Degenerate case: a perfectly flat baseline (std == 0). Any
        # deviation at all counts as out of limit.
        outside = value != mean
        distance = value - mean

    if outside:
        state['consecutive'] += 1
    else:
        state['consecutive'] = 0
        state['alerted'] = False

    is_anomaly = False
    if state['consecutive'] >= state['sustained_minutes'] and not state['alerted']:
        is_anomaly = True
        state['alerted'] = True

    return state, is_anomaly, distance


######
# Batch mode: run OOL on a full dataset, return list of anomaly dicts
######

def run_batch(features, timestamps, threshold=THRESHOLD, warmup=WARMUP,
             sustained_minutes=SUSTAINED_MINUTES):
    anomalies = []
    for name, values in features.items():
        state = ool_init(threshold, warmup, sustained_minutes)
        for i, value in enumerate(values):
            state, is_anomaly, distance = ool_update(state, value)
            if is_anomaly:
                # Collect all metrics at the anomaly timestamp
                metrics_at_time = {feat: float(feat_values[i])
                                   for feat, feat_values in features.items()}
                anomalies.append({
                    'timestamp': timestamps[i],
                    'feature': name,
                    'value': float(value),
                    'frozen_mean': state['frozen_mean'],
                    'frozen_std': state['frozen_std'],
                    'distance': distance,
                    'consecutive_minutes': state['consecutive'],
                    'metrics': metrics_at_time,
                })
    return anomalies


######
# Live monitor: poll CSV every minute, flag anomalies on new rows
######

def run_monitor(data_path=DATA_PATH, threshold=THRESHOLD, warmup=WARMUP,
                poll_interval=POLL_INTERVAL, use_db=USE_DB,
                conn_params=DB_CONN_PARAMS,
                sustained_minutes=SUSTAINED_MINUTES):
    features, _ = extract_all_features(data_path)
    # Now covers every feature, not just the percentage-based ones.
    states = {name: ool_init(threshold, warmup, sustained_minutes)
              for name in features.keys()}
    seen = set()

    # Anomalies are written to per-day CSV files (anomalies_YYYY-MM-DD.csv)
    # so every day gets its own file for easier inspection.
    anomalies_dir = os.path.join(_BASE_DIR, 'ANOMALIES')
    os.makedirs(anomalies_dir, exist_ok=True)
    logging.info(f"Anomalies will be written to {anomalies_dir}/anomalies_YYYY-MM-DD.csv")

    logging.info(f"Starting OOL monitor on {data_path}")
    logging.info(f"threshold={threshold} std, warmup={warmup}min, "
                 f"poll={poll_interval}s, sustained={sustained_minutes}min")
    if use_db:
        logging.info(f"DB fetch enabled: {conn_params['host']}:{conn_params['port']}/{conn_params['dbname']}")
    logging.info("Waiting for new minute data...")

    while True:
        if use_db:
            try:
                n = fetch_and_append(conn_params, data_path)
                if n:
                    logging.info(f"  fetched {n} row(s) from DB {time.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception as e:
                logging.error(f"  DB fetch failed: {e}")

        current_features, current_timestamps = extract_all_features(data_path)

        for i, ts in enumerate(current_timestamps):
            if ts in seen:
                continue
            seen.add(ts)
            for name, values in current_features.items():
                if name not in states:
                    continue
                value = values[i]
                states[name], is_anomaly, distance = ool_update(states[name], value)
                if is_anomaly:
                    mean = states[name]['frozen_mean']
                    std = states[name]['frozen_std']
                    # Collect all metrics at the anomaly timestamp
                    metrics_at_time = {feat: float(feat_values[i])
                                       for feat, feat_values in current_features.items()}
                    logging.warning(f"[{ts}] ANOMALY  {name}: value={value:.4f} "
                                    f"frozen_mean={mean:.4f} frozen_std={std:.4f} "
                                    f"(sustained {states[name]['consecutive']} min "
                                    f"beyond {threshold}std)")
                    daily_path = os.path.join(anomalies_dir, f"anomalies_{ts[:10]}.csv")
                    file_exists = os.path.exists(daily_path) and os.path.getsize(daily_path) > 0
                    with open(daily_path, "a", encoding="utf-8") as f:
                        if not file_exists:
                            f.write("timestamp,feature,value,frozen_mean,frozen_std,sustained,distance,metrics\n")
                        f.write(f"{ts},{name}: value={value:.4f}, "
                                f"frozen_mean={mean:.4f}, frozen_std={std:.4f}, "
                                f"sustained={states[name]['consecutive']}min, "
                                f"distance={distance:.2f}, "
                                f"metrics={metrics_at_time}\n")

        time.sleep(poll_interval)


if __name__ == "__main__":
    run_monitor()
