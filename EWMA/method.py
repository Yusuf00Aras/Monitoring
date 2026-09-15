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
THRESHOLD = 4.5
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
# EWMA state: {ewma, variance, n}
######

def ewma_init(alpha=ALPHA, threshold=THRESHOLD, warmup=WARMUP):
    return {'alpha': alpha, 'threshold': threshold, 'warmup': warmup,
            'ewma': None, 'variance': None, 'n': 0}


######
# Feed one value, return (state, is_anomaly, distance)
######

def ewma_update(state, value):
    value = float(value)
    alpha = state['alpha']
    threshold = state['threshold']
    warmup = state['warmup']

    if state['ewma'] is None:
        state['ewma'] = value
        state['variance'] = 0.0
        state['n'] = 1
        return state, False, 0.0

    state['ewma'] = alpha * value + (1 - alpha) * state['ewma']
    residual = value - state['ewma']
    std = np.sqrt(state['variance']) if state['variance'] > 0 else 0.0
    distance = abs(residual) / std if std > 0 else 0.0

    state['n'] += 1
    delta = residual - (residual / state['n'])
    state['variance'] += delta * residual / state['n']

    is_anomaly = state['n'] > warmup and distance > threshold
    return state, is_anomaly, distance


######
# Batch mode: run EWMA on a full dataset, return list of anomaly dicts
######

def run_batch(features, timestamps, alpha=ALPHA, threshold=THRESHOLD, warmup=WARMUP):
    anomalies = []
    for name, values in features.items():
        state = ewma_init(alpha, threshold, warmup)
        for i, value in enumerate(values):
            state, is_anomaly, distance = ewma_update(state, value)
            if is_anomaly:
                # Collect all metrics at the anomaly timestamp
                metrics_at_time = {feat: float(feat_values[i])
                                   for feat, feat_values in features.items()}
                anomalies.append({
                    'timestamp': timestamps[i],
                    'feature': name,
                    'value': float(value),
                    'ewma': state['ewma'],
                    'distance': distance,
                    'metrics': metrics_at_time,
                })
    return anomalies


######
# Live monitor: poll CSV every minute, flag anomalies on new rows
######

def run_monitor(data_path=DATA_PATH, alpha=ALPHA, threshold=THRESHOLD,
                poll_interval=POLL_INTERVAL, use_db=USE_DB,
                conn_params=DB_CONN_PARAMS):
    features, _ = extract_all_features(data_path)
    states = {name: ewma_init(alpha, threshold) for name in features.keys()}
    seen = set()

    # Create a fresh anomalies CSV (with timestamp in the name) so old runs stay untouched
    anomalies_path = os.path.join(_BASE_DIR, 'ANOMALIES',
                                 f"anomalies_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    os.makedirs(os.path.dirname(anomalies_path), exist_ok=True)
    with open(anomalies_path, "w", encoding="utf-8") as f:
        f.write("timestamp,feature,value,ewma,distance,metrics\n")
    logging.info(f"Anomalies will be written to {anomalies_path}")

    logging.info(f"Starting EWMA monitor on {data_path}")
    logging.info(f"alpha={alpha}, threshold={threshold}, poll={poll_interval}s")
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
                value = values[i]
                states[name], is_anomaly, distance = ewma_update(states[name], value)
                if is_anomaly:
                    # Collect all metrics at the anomaly timestamp
                    metrics_at_time = {feat: float(feat_values[i])
                                       for feat, feat_values in current_features.items()}
                    logging.warning(f"[{ts}] ANOMALY  {name}: value={value:.4f} "
                                    f"ewma={states[name]['ewma']:.4f} distance={distance:.2f}")
                    with open(anomalies_path, "a", encoding="utf-8") as f:
                        f.write(f"{ts},{name}: value={value:.4f}, "
                                f"ewma={states[name]['ewma']:.4f}, "
                                f"distance={distance:.2f}, "
                                f"metrics={metrics_at_time}\n")

        time.sleep(poll_interval)


if __name__ == "__main__":
    run_monitor()