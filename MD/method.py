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


# Squared Mahalanobis distance approximately follows a chi-square
# distribution with 11 degrees of freedom under multivariate normality.
# The 95th percentile is approximately 19.68.
#
# This implementation calculates the ordinary Mahalanobis distance D,
# therefore the corresponding threshold is:
# sqrt(19.68) ≈ 4.44.
THRESHOLD = 4.44

POLL_INTERVAL = 60


# Feature names in the exact order produced by extract_important_features()
FEATURE_NAMES = [
    "cpu_user_pct",
    "cpu_system_pct",
    "cpu_iowait_pct",
    "cpu_switches",
    "cpu_interrupts",
    "mem_util_pct",
    "mem_committed_as_kbytes",
    "sys_load_avg_1",
    "sys_proc_running",
    "sys_proc_count",
    "sys_swap_used_pct",
]


DB_CONN_PARAMS = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '5432'),
    'dbname': os.environ.get('DB_NAME', 'metrics'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', ''),
}

USE_DB = True


######
# Batch Mahalanobis distance using full-dataset covariance (unrelated to
# the online/frozen detector below -- kept only for ad-hoc inspection).
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
            inv_cov = np.linalg.inv(
                cov + regulator * np.eye(cov.shape[0])
            )
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)

    distances = []

    for x in scaled_data:
        distances.append(
            float(distance.mahalanobis(x, mean, inv_cov))
        )

    return distances


######
# Fit MD's frozen reference distribution (mean, inverse covariance) from
# a clean baseline window. Called once; the result is reused for scoring
# and is never updated afterward -- injected/evaluation data does not
# change the reference distribution.
######

def md_fit(baseline_vectors, regulator=1e-8):
    X = np.asarray(baseline_vectors, dtype=float)
    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    cov_reg = cov + regulator * np.eye(cov.shape[0])
    try:
        inv_cov = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov_reg)
    return {'mean': mean, 'inv_cov': inv_cov}


######
# MD state: frozen {mean, inv_cov, threshold} + running {in_alarm}
######

def md_init(mean, inv_cov, threshold=THRESHOLD):
    return {
        'mean': mean,
        'inv_cov': inv_cov,
        'threshold': threshold,
        'in_alarm': False,
    }


######
# Feed one vector, return (state, is_anomaly, distance).
# Scores against the frozen reference distribution only -- mean/inv_cov
# are never updated here. One alarm per excursion: is_anomaly is True only
# on the first minute above the threshold (same alarm unit as OOL/EWMA).
######

def md_update(state, x):
    x = np.asarray(x, dtype=float)
    diff = x - state['mean']
    dist = float(np.sqrt(diff @ state['inv_cov'] @ diff))
    above = dist > state['threshold']
    is_anomaly = above and not state['in_alarm']
    state['in_alarm'] = above
    return state, is_anomaly, dist


######
# Batch mode: fit the reference distribution once from baseline_vectors,
# then score every vector in `features` (which may include baseline and
# evaluation/injected data) against those fixed parameters.
######

def run_batch(baseline_vectors, features, timestamps, threshold=THRESHOLD, regulator=1e-8):
    fit = md_fit(baseline_vectors, regulator)
    state = md_init(fit['mean'], fit['inv_cov'], threshold)

    anomalies = []

    for i, x in enumerate(features):
        state, is_anomaly, dist = md_update(state, x)

        if is_anomaly:
            metrics_at_time = {
                name: float(x[j])
                for j, name in enumerate(FEATURE_NAMES)
                if j < len(x)
            }
            anomalies.append({
                'timestamp': timestamps[i],
                'distance': dist,
                'metrics': metrics_at_time,
            })

    return anomalies


######
# Live monitor: fits the reference distribution once from whatever
# historical data already exists in the CSV, then scores all subsequent
# (including newly polled) data against that fixed distribution.
######

def run_monitor(
    data_path=DATA_PATH,
    threshold=THRESHOLD,
    poll_interval=POLL_INTERVAL,
    use_db=USE_DB,
    conn_params=DB_CONN_PARAMS,
    regulator=1e-8,
):
    baseline_vectors, seen_timestamps = extract_important_features(data_path)
    seen = set(seen_timestamps)

    state = None
    if baseline_vectors:
        fit = md_fit(baseline_vectors, regulator)
        state = md_init(fit['mean'], fit['inv_cov'], threshold)
        logging.info(f"Baseline fitted from {len(baseline_vectors)} historical minutes; frozen.")
    else:
        logging.info("CSV is empty -- baseline will be fitted once enough data has accumulated.")

    anomalies_dir = os.path.join(_BASE_DIR, 'ANOMALIES')
    os.makedirs(anomalies_dir, exist_ok=True)
    logging.info(f"Anomalies will be written to {anomalies_dir}/anomalies_YYYY-MM-DD.csv")
    logging.info(f"Starting Mahalanobis monitor on {data_path}")
    logging.info(f"threshold={threshold}, poll={poll_interval}s (reference distribution frozen at fit time)")

    if use_db:
        logging.info(
            f"DB fetch enabled: {conn_params['host']}:{conn_params['port']}/{conn_params['dbname']}"
        )

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

        if state is None:
            if current_features:
                fit = md_fit(current_features, regulator)
                state = md_init(fit['mean'], fit['inv_cov'], threshold)
                seen = set(current_timestamps)
                logging.info(f"Baseline fitted from {len(current_features)} historical minutes; frozen.")
            time.sleep(poll_interval)
            continue

        for i, ts in enumerate(current_timestamps):
            if ts in seen:
                continue
            seen.add(ts)

            state, is_anomaly, dist = md_update(state, current_features[i])

            if is_anomaly:
                x = current_features[i]
                metrics_at_time = {
                    name: float(x[j])
                    for j, name in enumerate(FEATURE_NAMES)
                    if j < len(x)
                }
                logging.warning(
                    f"[{ts}] ANOMALY  Mahalanobis distance={dist:.4f} (threshold={threshold:.2f})"
                )
                daily_path = os.path.join(anomalies_dir, f"anomalies_{ts[:10]}.csv")
                file_exists = os.path.exists(daily_path) and os.path.getsize(daily_path) > 0
                with open(daily_path, "a", encoding="utf-8") as f:
                    if not file_exists:
                        f.write("timestamp,distance,metrics\n")
                    f.write(f"{ts},{dist:.4f}, metrics={metrics_at_time}\n")

        time.sleep(poll_interval)


if __name__ == "__main__":
    features, timestamps = extract_important_features(DATA_PATH)

    if features:
        distances = mahalanobis_distances(features)
        logging.info(
            f"Baseline: {len(distances)} minutes, "
            f"median distance={np.median(distances):.4f}, "
            f"max distance={max(distances):.4f}"
        )
    else:
        logging.info("CSV is empty -- starting fresh, baseline will be built from DB data.")

    run_monitor()