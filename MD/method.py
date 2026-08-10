from scipy.spatial import distance
import numpy as np
from cleaning_utils import extract_important_features

features = extract_important_features('./Test_Data/data-1786192670480.csv')

def mahalanobis_distances(data, regularize_eps=1e-8, use_pinv=True):
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    if use_pinv:
        inv_cov = np.linalg.pinv(cov)
    else:
        try:
            inv_cov = np.linalg.inv(cov + regularize_eps * np.eye(cov.shape[0]))
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)

    distances = np.array([distance.mahalanobis(x, mean, inv_cov) for x in data])
    return distances


# Module-level convenience variable (computed from `test_data`).
# Importing modules that rely on `distances` can use this variable.
distances = mahalanobis_distances(features)


if __name__ == "__main__":
    cov = np.cov(features, rowvar=False)
    distances = mahalanobis_distances(features)
    print("Covariance Matrix:", cov)
    print("Mahalanobis distances:", distances)
