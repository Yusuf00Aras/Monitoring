import numpy as np
from cleaning_utils import extract_important_features

data, time = extract_important_features('./Test_Data/data-1786192670480.csv')

def ewma(data):
    alpha= 0.3
    ewma_results = []

    for i in range(len(data)):
        if i == 0:
            ewma_results.append(data[i])
        else:
            ewma_results.append(alpha * data[i] + (1 - alpha) * ewma_results[i - 1])
    return ewma_results


if __name__ == "__main__":
    
    values = ewma(data)
    print(values[0:8])