import numpy as np
from cleaning_utils import extract_cpu_user_pct

cpu_user_pct = extract_cpu_user_pct('./Test_Data/data-1786192670480.csv')

def ewma(data, alpha=0.3):
    if not data:
        return []
    ewma_results = []
    ewma_results.append(data[0])

    for i in range(1, len(data)):
        ewma_value = alpha * data[i] + (1 - alpha) * ewma_results[i - 1]
        ewma_results.append(ewma_value)


    return ewma_results

if __name__ == "__main__":

    print(f"Original data[0:2]: {cpu_user_pct[0:2]}")
    values = ewma(cpu_user_pct)
    print(f"EWMA values[0:2]: {values[0:2]}")  