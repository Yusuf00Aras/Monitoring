import os
import time
from scipy.spatial import distance
import numpy as np
from cleaning_utils import extract_important_features
from sklearn.preprocessing import StandardScaler
from db_utils import fetch_and_append

# Resolve the CSV path relative to this file so the script works no matter
# what the current working directory is (important on the server).
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_BASE_DIR, '..', 'Test_Data', 'data-1786192670480.csv')

THRESHOLD = 3.0       # Mahalanobis distance above which a minute is flagged
POLL_INTERVAL = 60    # seconds between CSV checks
WARMUP = 10          # minutes to collect before anomaly detection kicks in

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


features, time = extract_important_features(DATA_PATH)
# debugging print(time[0])
# debugging print(len(features), len(time))
# debugging print(np.shape(features)), np.array with 1439 rows and 11 columns, each row is a minute


# regulator is a small value added to the covariance matrix so the determinant is not zero, which makes the matrix invertible
def mahalanobis_distances(data, regulator =1e-8, invertible=True):

    # difference in measurements could influuence the outcome thats why we scale the data
    scale = StandardScaler()
    scaled_data = scale.fit_transform(data)
    # axis 0 means we take each element of a row as a feature and calculate the mean for each feature
    mean = np.mean(scaled_data, axis=0) 
    # each column is a feature thats why rowvar=false else we would get the covariance of each row
    cov = np.cov(scaled_data, rowvar=False)


    # if the covariance matrix is invertible we calculate the inverse of the covariance matrix
    if invertible:
        inv_cov = np.linalg.pinv(cov) 
    # if it is not invertible we add small values to the diagnoal of the covariance matrix to make it invertible
    else:
        try:
            # np eye returns an identity matrix with the same dimensions and then we multiply each diagonal element with the regulator value and add it to the covariance matrix
            # this creates a invertible covariance matrix
            inv_cov = np.linalg.inv(cov + regulator * np.eye(cov.shape[0]))
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)

    # calculate the mahalanobis distance for each row in the data using the mean and inverse covariance matrix
    distances = []
    for x in scaled_data:
        distances.append(float(distance.mahalanobis(x, mean, inv_cov)))



    return distances


class MahalanobisMonitor:
    """Incrementally tracks all features together and flags minutes whose
    multivariate Mahalanobis distance from the running mean is too large.

    Uses Welford's online algorithm to update the mean and covariance matrix
    one observation at a time, so it can process a stream of minute vectors
    without re-scanning the whole history.
    """

    def __init__(self, threshold=THRESHOLD, warmup=WARMUP, regulator=1e-8):
        self.threshold = threshold
        self.warmup = warmup
        self.regulator = regulator
        self.n = 0                  # number of observations seen
        self.mean = None            # running mean vector
        self.cov = None             # running covariance matrix (M2 / (n-1))

    def update(self, x):
        """Feed a new minute vector to the monitor.

        Returns (is_anomaly, dist) where dist is the Mahalanobis distance
        of the new vector from the running mean.
        """
        x = np.asarray(x, dtype=float)

        # first observation seeds the mean, nothing to compare against yet
        if self.n == 0:
            self.mean = x.copy()
            self.cov = np.zeros((len(x), len(x)))
            self.n = 1
            return False, 0.0

        # --- Welford online update for mean and covariance ---
        n = self.n + 1
        delta = x - self.mean               # (x - mean_old)
        self.mean += delta / n               # mean_new
        delta2 = x - self.mean               # (x - mean_new)
        # outer product update of the co-moment matrix M2
        self.cov = ((self.cov * (self.n - 1)) + np.outer(delta, delta2)) / (n - 1)
        self.n = n

        # only flag anomalies once enough data has been collected for a
        # stable covariance estimate
        if self.n <= self.warmup:
            return False, 0.0

        # --- Mahalanobis distance with the running statistics ---
        # add a small regulator so the covariance matrix is always invertible
        cov = self.cov + self.regulator * np.eye(self.cov.shape[0])
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)

        diff = x - self.mean
        dist = float(np.sqrt(diff @ inv_cov @ diff))

        is_anomaly = dist > self.threshold
        return is_anomaly, dist


def run_monitor(data_path=DATA_PATH, threshold=THRESHOLD,
                poll_interval=POLL_INTERVAL, use_db=USE_DB,
                conn_params=DB_CONN_PARAMS):
    """Poll the CSV every minute, feed new rows into the Mahalanobis
    monitor, and print whenever a minute drifts too far from the norm.

    When use_db is True, the last completed minute is fetched from the
    database and appended to the CSV before each cycle, so the monitor
    keeps running continuously on the AWS server.
    """
    monitor = MahalanobisMonitor(threshold=threshold)
    seen = set()  # timestamps already processed

    print(f"Starting Mahalanobis monitor on {data_path}")
    print(f"threshold={threshold}, poll={poll_interval}s, warmup={monitor.warmup}")
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
        current_features, current_timestamps = extract_important_features(data_path)

        for i, ts in enumerate(current_timestamps):
            if ts in seen:
                continue

            seen.add(ts)
            is_anomaly, dist = monitor.update(current_features[i])

            if is_anomaly:
                print(
                    f"[{ts}] ANOMALY  Mahalanobis distance={dist:.4f} "
                    f"(threshold={threshold})"
                )

        # wait until the next minute
        time.sleep(poll_interval)


if __name__ == "__main__":
    distances = mahalanobis_distances(features)
    print(distances[577]) 
    

    #++++++++++++++++++++++++++++++++++++++++++++++++#
    # debugging print(distances[0]) # mahalanobis distance for the first minute and all the features pertaining to that minute
    # debugging print(type(distances)) (np array)
    # debugging print(distances[0:5]), distances is one dimensional array with length equal to the number of rows in features (minutes)
    # debugging print("Mahalanobis distances:", len(distances)), number of minutes

    run_monitor()