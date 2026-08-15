import numpy as np
from cleaning_utils import extract_important_features

data, time = extract_important_features('./Test_Data/data-1786192670480.csv')

def ewma(data, alpha=0.3):
    if not data:
        return []

    ewma_results = data[0]


    for i in range(1, len(data)):
        current_row = data[i]
        prev_ewma_row = ewma_results[i - 1]
        
        smoothed_row = [
            alpha * current_val + (1 - alpha) * prev_val
            for current_val, prev_val in zip(current_row, prev_ewma_row) ]
        
        ewma_results.append(smoothed_row)

    return ewma_results

if __name__ == "__main__":

    print(f"Original data[0:2]: {data[0:2]}")
    values = ewma(data)
    print(f"EWMA values[0:2]: {values[0:2]}")  