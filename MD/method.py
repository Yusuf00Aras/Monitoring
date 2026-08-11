from scipy.spatial import distance
import numpy as np
from cleaning_utils import extract_important_features

features = extract_important_features('./Test_Data/data-1786192670480.csv')
# debugging print(features[0])
# debugging print(np.shape(features)), np.array with 1439 rows and 11 columns, each row is a minute


# regulator is a small value added to the covariance matrix so the determinant is not zero, which makes the matrix invertible
def mahalanobis_distances(data, regulator =1e-8, invertible=True):
    # axis 0 means we take each element of a row as a feature and calculate the mean for each feature
    mean = np.mean(data, axis=0) 
    # each column is a feature thats why rowvar=false else we would get the covariance of each row
    cov = np.cov(data, rowvar=False)


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
    # returning an array of distances for each minute
    distances = np.array([distance.mahalanobis(x, mean, inv_cov) for x in data])
    return distances



if __name__ == "__main__":
    cov = np.cov(features, rowvar=False)
    distances = mahalanobis_distances(features)
    print(distances)

    #++++++++++++++++++++++++++++++++++++++++++++++++#
    # debugging print(distances[0]) # mahalanobis distance for the first minute and all the features pertaining to that minute
    # debugging print(type(distances)) (np array)
    # debugging print(distances[0:5]), distances is one dimensional array with length equal to the number of rows in features (minutes)
    # debugging print("Mahalanobis distances:", len(distances)), number of minutes
