"""EWMA control chart (Roberts, 1959) -- one chart per metric.

The EWMA statistic z_t = alpha * x_t + (1 - alpha) * z_{t-1} is compared
against a FROZEN reference (mean mu_0, standard deviation sigma_0) that is
fitted once on a clean baseline window, exactly like MD's reference
distribution. An alarm is raised when

    |z_t - mu_0| > L * sigma_z(t),
    sigma_z(t) = sigma_0 * sqrt(alpha / (2 - alpha) * (1 - (1 - alpha)^(2t)))

Because the reference is frozen, a slow drift pushes z_t steadily away
from mu_0 instead of being absorbed by the chart. One alarm is raised per
excursion (the first minute above the limit), the same alarm unit as OOL
and MD.
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

ALPHA = 0.3
# Control-limit width L in units of sigma_z (classical value L = 3).
THRESHOLD = 3.0
POLL_INTERVAL = 60

# Minimum number of historical minutes before the frozen reference is
# fitted (12 h, same as the evaluation baseline window). Until then the
# monitor only collects data.
MIN_BASELINE_MINUTES = 720

# Number of alarms the threshold is calibrated to on the clean baseline
# (same budget as in the evaluation). The live monitor calibrates its
# threshold automatically unless one is passed explicitly.
CALIBRATION_TARGET = 1

DB_CONN_PARAMS = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '5432'),
    'dbname': os.environ.get('DB_NAME', 'metrics'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', ''),
}

USE_DB = True


######
# Fit the frozen reference (mu_0, sigma_0) of one metric from a clean
# baseline window. Called once; never updated afterward.
######

def ewma_fit(baseline_values):
    x = np.asarray(baseline_values, dtype=float)
    return {'mean': float(np.mean(x)), 'std': float(np.std(x, ddof=1)) if len(x) > 1 else 0.0}


######
# EWMA state: frozen {mean, std} + running {z, t, in_alarm}
######

def ewma_init(mean, std, alpha=ALPHA, threshold=THRESHOLD):
    return {'alpha': alpha, 'threshold': threshold,
            'mean': float(mean), 'std': float(std),
            'z': float(mean), 't': 0, 'in_alarm': False}


######
# Feed one value, return (state, is_anomaly, distance).
# distance = |z_t - mu_0| / sigma_z(t); is_anomaly only on the first
# minute of an excursion above L.
######

def ewma_update(state, value):
    alpha = state['alpha']
    state['t'] += 1
    state['z'] = alpha * float(value) + (1 - alpha) * state['z']

    sigma_z = state['std'] * np.sqrt(
        alpha / (2 - alpha) * (1 - (1 - alpha) ** (2 * state['t'])))
    distance = abs(state['z'] - state['mean']) / sigma_z if sigma_z > 0 else 0.0

    above = distance > state['threshold']
    is_anomaly = above and not state['in_alarm']
    state['in_alarm'] = above
    return state, is_anomaly, distance


######
# Fit one chart per metric from a baseline dict {feature: [values]}
######

def fit_states(baseline_features, alpha=ALPHA, threshold=THRESHOLD):
    states = {}
    for name, values in baseline_features.items():
        fit = ewma_fit(values)
        states[name] = ewma_init(fit['mean'], fit['std'], alpha, threshold)
    return states


######
# Batch mode: fit the frozen reference from baseline_features, then run the
# charts over `features` (which may include baseline and evaluation data).
######

def run_batch(baseline_features, features, timestamps, alpha=ALPHA, threshold=THRESHOLD):
    states = fit_states(baseline_features, alpha, threshold)
    anomalies = []
    for name, values in features.items():
        state = states[name]
        for i, value in enumerate(values):
            state, is_anomaly, distance = ewma_update(state, value)
            if is_anomaly:
                metrics_at_time = {feat: float(feat_values[i])
                                   for feat, feat_values in features.items()}
                anomalies.append({
                    'timestamp': timestamps[i],
                    'feature': name,
                    'value': float(value),
                    'ewma': state['z'],
                    'distance': distance,
                    'metrics': metrics_at_time,
                })
    return anomalies


######
# Calibrate L on a clean baseline: binary search for the smallest L that
# produces at most `target` alarms when the charts are fitted on and run
# over the baseline itself.
######

def calibrate_threshold(baseline_features, timestamps, target=CALIBRATION_TARGET,
                        alpha=ALPHA, low=0.5, high=100.0, tolerance=0.01, max_iter=50):
    best = high
    for _ in range(max_iter):
        mid = (low + high) / 2
        n_alarms = len(run_batch(baseline_features, baseline_features, timestamps,
                                 alpha=alpha, threshold=mid))
        if n_alarms <= target:
            best = high = mid
        else:
            low = mid
        if high - low < tolerance:
            break
    return best


######
# Fit the frozen reference (and calibrate L if threshold is None).
######

def _fit_baseline(baseline_features, timestamps, alpha, threshold):
    if threshold is None:
        threshold = calibrate_threshold(baseline_features, timestamps, alpha=alpha)
        logging.info(f"L calibrated on baseline to {threshold:.4f} "
                     f"({CALIBRATION_TARGET} baseline alarm(s))")
    logging.info(f"Baseline fitted from {len(timestamps)} historical minutes; frozen.")
    return fit_states(baseline_features, alpha, threshold), threshold


######
# Live monitor: fits the reference once from the historical data already
# in the CSV, then charts all subsequent (newly polled) minutes.
# threshold=None calibrates L automatically on that baseline.
######

def run_monitor(data_path=DATA_PATH, alpha=ALPHA, threshold=None,
                poll_interval=POLL_INTERVAL, use_db=USE_DB,
                conn_params=DB_CONN_PARAMS):
    baseline_features, seen_timestamps = extract_all_features(data_path)
    seen = set(seen_timestamps)

    states = None
    if len(seen_timestamps) >= MIN_BASELINE_MINUTES:
        states, threshold = _fit_baseline(baseline_features, seen_timestamps, alpha, threshold)
    else:
        logging.info(f"{len(seen_timestamps)} historical minutes -- baseline will be fitted "
                     f"once {MIN_BASELINE_MINUTES} minutes have accumulated.")

    # Anomalies are written to per-day CSV files (anomalies_YYYY-MM-DD.csv)
    # so every day gets its own file for easier inspection.
    anomalies_dir = os.path.join(_BASE_DIR, 'ANOMALIES')
    os.makedirs(anomalies_dir, exist_ok=True)
    logging.info(f"Anomalies will be written to {anomalies_dir}/anomalies_YYYY-MM-DD.csv")

    logging.info(f"Starting EWMA control chart monitor on {data_path}")
    logging.info(f"alpha={alpha}, L={'auto' if threshold is None else threshold}, "
                 f"poll={poll_interval}s")
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

        if states is None:
            if len(current_timestamps) >= MIN_BASELINE_MINUTES:
                states, threshold = _fit_baseline(current_features, current_timestamps,
                                                  alpha, threshold)
                seen = set(current_timestamps)
            time.sleep(poll_interval)
            continue

        for i, ts in enumerate(current_timestamps):
            if ts in seen:
                continue
            seen.add(ts)
            for name, values in current_features.items():
                value = values[i]
                states[name], is_anomaly, distance = ewma_update(states[name], value)
                if is_anomaly:
                    # Collect all metrics at the anomaly timestamp
                    metrics_at_time = {feat: float(feat_values[i])
                                       for feat, feat_values in current_features.items()}
                    z = states[name]['z']
                    logging.warning(f"[{ts}] ANOMALY  {name}: value={value:.4f} "
                                    f"ewma={z:.4f} distance={distance:.2f}")
                    daily_path = os.path.join(anomalies_dir, f"anomalies_{ts[:10]}.csv")
                    file_exists = os.path.exists(daily_path) and os.path.getsize(daily_path) > 0
                    with open(daily_path, "a", encoding="utf-8") as f:
                        if not file_exists:
                            f.write("timestamp,feature,value,ewma,distance,metrics\n")
                        f.write(f"{ts},{name}: value={value:.4f}, "
                                f"ewma={z:.4f}, "
                                f"distance={distance:.2f}, "
                                f"metrics={metrics_at_time}\n")

        time.sleep(poll_interval)


if __name__ == "__main__":
    run_monitor()
