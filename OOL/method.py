"""Out-of-Limit (OOL) anomaly detection -- STATIC thresholds.

Mirrors the current production monitoring: every monitored metric has a
fixed, hand-configured upper limit. A value is out of limit when it
exceeds that limit, and an anomaly is raised once the metric has stayed
out of limit for SUSTAINED_MINUTES consecutive minutes (one alert per
excursion).

The limits are NOT learned from data and NOT calibrated. OOL is the
unchanged reference that EWMA and MD are compared against.
"""
import os
import time
import logging
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

# Static upper limits, identical to the production Zabbix triggers.
# Only metrics listed here are monitored by OOL (as in production).
# TODO: replace with the exact production trigger values.
LIMITS = {
    "cpu_user_pct": 90.0,
    "cpu_system_pct": 90.0,
    "cpu_iowait_pct": 90.0,
    "mem_util_pct": 90.0,
    "sys_swap_used_pct": 90.0,
}

# Metric must stay above its limit this many consecutive minutes.
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
# OOL state: {limit, sustained_minutes, consecutive, alerted}
######

def ool_init(limit, sustained_minutes=SUSTAINED_MINUTES):
    return {
        'limit': float(limit),
        'sustained_minutes': sustained_minutes,
        'consecutive': 0,
        'alerted': False,
    }


######
# Feed one value, return (state, is_anomaly, distance).
# distance = value - limit (positive means above the limit).
######

def ool_update(state, value):
    value = float(value)
    distance = value - state['limit']

    if value > state['limit']:
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
# Batch mode: run OOL on a full dataset, return list of anomaly dicts.
# `threshold` is accepted and ignored so the evaluation interface stays
# uniform -- OOL limits are static and never tuned.
######

def run_batch(features, timestamps, limits=None,
              sustained_minutes=SUSTAINED_MINUTES, threshold=None, **_ignored):
    limits = LIMITS if limits is None else limits
    anomalies = []
    for name, limit in limits.items():
        if name not in features:
            continue
        state = ool_init(limit, sustained_minutes)
        for i, value in enumerate(features[name]):
            state, is_anomaly, distance = ool_update(state, value)
            if is_anomaly:
                metrics_at_time = {feat: float(vals[i]) for feat, vals in features.items()}
                anomalies.append({
                    'timestamp': timestamps[i],
                    'feature': name,
                    'value': float(value),
                    'limit': float(limit),
                    'distance': distance,
                    'consecutive_minutes': state['consecutive'],
                    'metrics': metrics_at_time,
                })
    return anomalies


######
# Live monitor: poll CSV every minute, flag anomalies on new rows
######

def run_monitor(data_path=DATA_PATH, limits=None, poll_interval=POLL_INTERVAL,
                use_db=USE_DB, conn_params=DB_CONN_PARAMS,
                sustained_minutes=SUSTAINED_MINUTES):
    limits = LIMITS if limits is None else limits
    states = {name: ool_init(limit, sustained_minutes) for name, limit in limits.items()}
    seen = set()

    anomalies_dir = os.path.join(_BASE_DIR, 'ANOMALIES')
    os.makedirs(anomalies_dir, exist_ok=True)
    logging.info(f"Anomalies will be written to {anomalies_dir}/anomalies_YYYY-MM-DD.csv")
    logging.info(f"Starting static OOL monitor on {data_path}")
    logging.info(f"limits={limits}, sustained={sustained_minutes}min, poll={poll_interval}s")
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
            for name, state in states.items():
                value = current_features[name][i]
                states[name], is_anomaly, distance = ool_update(state, value)
                if is_anomaly:
                    limit = states[name]['limit']
                    consecutive = states[name]['consecutive']
                    metrics_at_time = {feat: float(vals[i]) for feat, vals in current_features.items()}
                    logging.warning(f"[{ts}] ANOMALY  {name}: value={value:.4f} "
                                    f"limit={limit:.2f} (sustained {consecutive} min)")
                    daily_path = os.path.join(anomalies_dir, f"anomalies_{ts[:10]}.csv")
                    file_exists = os.path.exists(daily_path) and os.path.getsize(daily_path) > 0
                    with open(daily_path, "a", encoding="utf-8") as f:
                        if not file_exists:
                            f.write("timestamp,feature,value,limit,sustained,metrics\n")
                        f.write(f"{ts},{name},{value:.4f},{limit:.2f},"
                                f"{consecutive},\"{metrics_at_time}\"\n")

        time.sleep(poll_interval)


if __name__ == "__main__":
    run_monitor()
