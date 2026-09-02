import os
import time
import numpy as np
import pandas as pd
from cleaning_utils import extract_all_features
from db_utils import fetch_and_append

# Resolve the CSV path relative to this file so the script works no matter
# what the current working directory is (important on the server).
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_BASE_DIR, '..', 'Test_Data', 'data-1786192670480.csv')
ALPHA = 0.3
# how many standard deviations a value may be away from the EWMA before it is flagged
THRESHOLD = 3.0
POLL_INTERVAL = 60  # seconds between CSV checks
# number of observations to collect before anomaly detection kicks in,
# so the running variance has time to stabilise (avoids cold-start false positives)
WARMUP = 10

# Database connection parameters for the AWS server.
# Override via environment variables so secrets are not hard-coded.
DB_CONN_PARAMS = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '5432'),
    'dbname': os.environ.get('DB_NAME', 'metrics'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', ''),
}

# Set to False to disable the DB fetch and just re-read the CSV (e.g. for
# local testing without a database connection).
USE_DB = True


features, timestamps = extract_all_features(DATA_PATH)

df = pd.DataFrame(features)

ewm_test = df.ewm(alpha=ALPHA, adjust=False).mean() # adjust = false because we want to use the recursive formula for EWMA, which is more efficient for large datasets.


# def ewma(data, alpha=0.3):  
# 
#     if not data:
#         return []
#     ewma_results = []
#     ewma_results.append(data[0])
# 
#     for i in range(1, len(data)):
#         ewma_value = alpha * data[i] + (1 - alpha) * ewma_results[i - 1]
#         ewma_results.append(ewma_value)
# 
# 
#     return ewma_results


class EwmaMonitor:
    """Incrementally tracks one feature with a recursive EWMA and flags
    values that drift too far from it.

    Uses the recursive formula already present in this file:
        ewma_t = alpha * x_t + (1 - alpha) * ewma_{t-1}
    """

    def __init__(self, alpha=ALPHA, threshold=THRESHOLD, warmup=WARMUP):
        self.alpha = alpha
        self.threshold = threshold
        self.warmup = warmup
        self.ewma = None          # current EWMA value
        self.variance = None       # running variance of the residuals
        self.n = 0                 # number of observations seen

    def update(self, value):
        """Feed a new minute value to the monitor.

        Returns (is_anomaly, distance) where distance is how many
        standard deviations the value is away from the EWMA.
        """
        value = float(value)

        # first observation seeds the EWMA, nothing to compare against yet
        if self.ewma is None:
            self.ewma = value
            self.variance = 0.0
            self.n = 1
            return False, 0.0

        # recursive EWMA update
        self.ewma = self.alpha * value + (1 - self.alpha) * self.ewma

        # distance from the EWMA, measured in running standard deviations
        residual = value - self.ewma
        std = np.sqrt(self.variance) if self.variance > 0 else 0.0
        distance = abs(residual) / std if std > 0 else 0.0

        # update the running variance of the residuals (Welford-style)
        self.n += 1
        delta = residual - (residual / self.n)
        self.variance += delta * residual / self.n

        # only flag anomalies once the variance has had time to stabilise
        is_anomaly = self.n > self.warmup and distance > self.threshold
        return is_anomaly, distance


def run_monitor(data_path=DATA_PATH, alpha=ALPHA, threshold=THRESHOLD,
                poll_interval=POLL_INTERVAL, use_db=USE_DB,
                conn_params=DB_CONN_PARAMS):
    """Poll the CSV every minute, feed new rows into per-feature EWMA
    monitors, and print whenever a value drifts too far from its EWMA.

    When use_db is True, the last completed minute is fetched from the
    database and appended to the CSV before each cycle, so the monitor
    keeps running continuously on the AWS server.
    """
    # one monitor per feature
    monitors = {name: EwmaMonitor(alpha, threshold) for name in features.keys()}
    seen = set()  # timestamps already processed

    print(f"Starting EWMA monitor on {data_path}")
    print(f"alpha={alpha}, threshold={threshold} std devs, poll={poll_interval}s")
    if use_db:
        print(f"DB fetch enabled: {conn_params['host']}:{conn_params['port']}/{conn_params['dbname']}")
    print("Waiting for new minute data...\n")

    while True:
        # fetch the last completed minute from the DB and append it to the CSV
        if use_db:
            try:
                n = fetch_and_append(conn_params, data_path)
                if n:
                    print(f"  fetched {n} row(s) from DB")
            except Exception as e:
                print(f"  DB fetch failed: {e}")

        # re-read the CSV each cycle so newly appended rows are picked up
        current_features, current_timestamps = extract_all_features(data_path)

        for i, ts in enumerate(current_timestamps):
            if ts in seen:
                continue

            seen.add(ts)
            for name, values in current_features.items():
                value = values[i]
                is_anomaly, distance = monitors[name].update(value)

                if is_anomaly:
                    print(
                        f"[{ts}] ANOMALY  {name}: value={value:.4f} "
                        f"ewma={monitors[name].ewma:.4f} "
                        f"distance={distance:.2f} std devs"
                    )

        # wait until the next minute
        time.sleep(poll_interval)


if __name__ == "__main__":

    #print(f"Original cpu_user_pct[0:10]: {features['cpu_user_pct'][0:10]}")
    #values = ewma(cpu_user_pct)
    #print(f"EWMA ewm values[0:10]: {ewm['cpu_user_pct'].values[0:10]}")
    print("_-------------------------------------------_")
    #print(f"EWMA:{float(ewm_test.values[0:10])}")
    print("_-------------------------------------------_")

    run_monitor()