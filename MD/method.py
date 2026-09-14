import os
import time
import logging
from scipy.spatial import distance
import numpy as np
from cleaning_utils import extract_important_features
from sklearn.preprocessing import StandardScaler
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

THRESHOLD = 7.0
POLL_INTERVAL = 60
WARMUP = 100

DB_CONN_PARAMS = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '5432'),
    'dbname': os.environ.get('DB_NAME', 'metrics'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', ''),
}

USE_DB = True


######
# Batch Mahalanobis distance using full-dataset covariance (scipy)
######

def mahalanobis_distances(data, regulator=1e-8, invertible=True):
    scale = StandardScaler()
    scaled_data = scale.fit_transform(data)
    mean = np.mean(scaled_data, axis=0)
    cov = np.cov(scaled_data, rowvar=False)

    if invertible:
        inv_cov = np.linalg.pinv(cov)
    else:
        try:
            inv_cov = np.linalg.inv(cov + regulator * np.eye(cov.shape[0]))
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)

    distances = []
    for x in scaled_data:
        distances.append(float(distance.mahalanobis(x, mean, inv_cov)))
    return distances


######
# MD state: {n, mean, cov}
######

def md_init(threshold=THRESHOLD, warmup=WARMUP, regulator=1e-8):
    return {'threshold': threshold, 'warmup': warmup, 'regulator': regulator,
            'n': 0, 'mean': None, 'cov': None}


######
# Feed one vector, return (state, is_anomaly, distance) via Welford online update
######

def md_update(state, x):
    x = np.asarray(x, dtype=float)
    threshold = state['threshold']
    warmup = state['warmup']
    regulator = state['regulator']

    if state['n'] == 0:
        state['mean'] = x.copy()
        state['cov'] = np.zeros((len(x), len(x)))
        state['n'] = 1
        return state, False, 0.0

    n = state['n'] + 1
    delta = x - state['mean']
    state['mean'] += delta / n
    delta2 = x - state['mean']
    state['cov'] = ((state['cov'] * (state['n'] - 1)) + np.outer(delta, delta2)) / (n - 1)
    state['n'] = n

    if state['n'] <= warmup:
        return state, False, 0.0

    cov = state['cov'] + regulator * np.eye(state['cov'].shape[0])
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)

    diff = x - state['mean']
    dist = float(np.sqrt(diff @ inv_cov @ diff))
    is_anomaly = dist > threshold
    return state, is_anomaly, dist


######
# Batch mode: run MD on a full dataset, return list of anomaly dicts
######

def run_batch(features, timestamps, threshold=THRESHOLD, warmup=WARMUP, regulator=1e-8):
    state = md_init(threshold, warmup, regulator)
    anomalies = []
    for i, x in enumerate(features):
        state, is_anomaly, dist = md_update(state, x)
        if is_anomaly:
            anomalies.append({'timestamp': timestamps[i], 'distance': dist})
    return anomalies


######
# Live monitor: poll CSV every minute, flag anomalies on new rows
######

def run_monitor(data_path=DATA_PATH, threshold=THRESHOLD,
                poll_interval=POLL_INTERVAL, use_db=USE_DB,
                conn_params=DB_CONN_PARAMS):
    state = md_init(threshold)
    seen = set()

    logging.info(f"Starting Mahalanobis monitor on {data_path}")
    logging.info(f"threshold={threshold}, poll={poll_interval}s, warmup={state['warmup']}")
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

        current_features, current_timestamps = extract_important_features(data_path)

        for i, ts in enumerate(current_timestamps):
            if ts in seen:
                continue
            seen.add(ts)
            state, is_anomaly, dist = md_update(state, current_features[i])
            if is_anomaly:
                logging.warning(f"[{ts}] ANOMALY  Mahalanobis distance={dist:.4f} "
                                f"(threshold={threshold})")
                with open("anomalies.csv", "a", encoding="utf-8") as f:
                    f.write(f"{ts},{dist:.4f}\n")

        time.sleep(poll_interval)


if __name__ == "__main__":
    features, timestamps = extract_important_features(DATA_PATH)
    if features:
        distances = mahalanobis_distances(features)
        logging.info(f"Baseline: {len(distances)} minutes, "
                     f"median distance={np.median(distances):.4f}, "
                     f"max distance={max(distances):.4f}")
    else:
        logging.info("CSV is empty — starting fresh, baseline will be built from DB data.")
    run_monitor()