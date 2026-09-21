"""Out-of-Limit (OOL) anomaly detection method.

The OOL method uses a simple static threshold: if a metric stays above
a fixed limit (e.g. 90% utilization) for a sustained number of minutes,
it is flagged as an anomaly.  No baseline calculation, no mean/std —
just a hard limit with a duration requirement.
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

POLL_INTERVAL = 60

# Static limits per feature. Only percentage-based features get a 90% limit;
# non-percentage features are skipped (no static threshold makes sense).
STATIC_LIMITS = {
    "cpu_user_pct": 90.0,
    "cpu_system_pct": 90.0,
    "cpu_iowait_pct": 90.0,
    "mem_util_pct": 90.0,
    "sys_swap_used_pct": 90.0,
}

# An anomaly is flagged after the value stays above the static limit
# for this many consecutive minutes.
SUSTAINED_MINUTES = 5

DB_CONN_PARAMS = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '5432'),
    'dbname': os.environ.get('DB_NAME', 'metrics'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', ''),
}

USE_DB = True


######
# OOL state: {limit, consecutive, alerted}
######

def ool_init(limit, sustained_minutes=SUSTAINED_MINUTES):
    return {'limit': limit, 'sustained_minutes': sustained_minutes,
            'consecutive': 0, 'alerted': False}


######
# Feed one value, return (state, is_anomaly, distance).
# An anomaly is flagged after the value stays above the static limit
# for sustained_minutes consecutive minutes.
######

def ool_update(state, value):
    value = float(value)
    limit = state['limit']
    sustained_minutes = state['sustained_minutes']

    if value > limit:
        state['consecutive'] += 1
    else:
        state['consecutive'] = 0
        state['alerted'] = False

    is_anomaly = False
    if state['consecutive'] >= sustained_minutes and not state['alerted']:
        is_anomaly = True
        state['alerted'] = True

    distance = value - limit
    return state, is_anomaly, distance


######
# Batch mode: run OOL on a full dataset, return list of anomaly dicts
######

def run_batch(features, timestamps, sustained_minutes=SUSTAINED_MINUTES):
    anomalies = []
    for name, values in features.items():
        if name not in STATIC_LIMITS:
            continue
        state = ool_init(STATIC_LIMITS[name], sustained_minutes)
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
                    'limit': state['limit'],
                    'distance': distance,
                    'consecutive_minutes': state['consecutive'],
                    'metrics': metrics_at_time,
                })
    return anomalies


######
# Live monitor: poll CSV every minute, flag anomalies on new rows
######

def run_monitor(data_path=DATA_PATH,
                poll_interval=POLL_INTERVAL, use_db=USE_DB,
                conn_params=DB_CONN_PARAMS,
                sustained_minutes=SUSTAINED_MINUTES):
    features, _ = extract_all_features(data_path)
    # Only monitor features that have a static limit defined
    states = {name: ool_init(STATIC_LIMITS[name], sustained_minutes)
              for name in features.keys() if name in STATIC_LIMITS}
    seen = set()

    # Anomalies are written to per-day CSV files (anomalies_YYYY-MM-DD.csv)
    # so every day gets its own file for easier inspection.
    anomalies_dir = os.path.join(_BASE_DIR, 'ANOMALIES')
    os.makedirs(anomalies_dir, exist_ok=True)
    logging.info(f"Anomalies will be written to {anomalies_dir}/anomalies_YYYY-MM-DD.csv")

    logging.info(f"Starting OOL monitor on {data_path}")
    logging.info(f"static limits={STATIC_LIMITS}, poll={poll_interval}s, "
                 f"sustained={sustained_minutes}min")
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
                    limit = states[name]['limit']
                    # Collect all metrics at the anomaly timestamp
                    metrics_at_time = {feat: float(feat_values[i])
                                       for feat, feat_values in current_features.items()}
                    logging.warning(f"[{ts}] ANOMALY  {name}: value={value:.4f} "
                                    f"limit={limit:.1f}% "
                                    f"(sustained {states[name]['consecutive']} min "
                                    f"over {limit:.1f}%)")
                    daily_path = os.path.join(anomalies_dir, f"anomalies_{ts[:10]}.csv")
                    file_exists = os.path.exists(daily_path) and os.path.getsize(daily_path) > 0
                    with open(daily_path, "a", encoding="utf-8") as f:
                        if not file_exists:
                            f.write("timestamp,feature,value,limit,sustained,distance,metrics\n")
                        f.write(f"{ts},{name}: value={value:.4f}, "
                                f"limit={limit:.1f}%, "
                                f"sustained={states[name]['consecutive']}min, "
                                f"distance={distance:.2f}, "
                                f"metrics={metrics_at_time}\n")

        time.sleep(poll_interval)


if __name__ == "__main__":
    run_monitor()
