import pandas as pd
import numpy as np
from cleaning_utils import extract_cpu_user_pct, extract_cpu_system_pct

cpu_user_pct = extract_cpu_user_pct('./Test_Data/data-1786192670480.csv')
cpu_system_pct = extract_cpu_system_pct('./Test_Data/data-1786192670480.csv')


dict = {
    "cpu_user_pct": cpu_user_pct,
    "cpu_system_pct": cpu_system_pct
}

df = pd.DataFrame(dict)
print(type(df))
ewm = df.ewm(alpha=0.3).mean() # mean allows average, else just a window object is returned.
ewm_test = df.ewm(alpha=0.3, adjust=False).mean() # adjust = false because we want to use the recursive formula for EWMA, which is more efficient for large datasets.

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


if __name__ == "__main__":

    print(f"Original cpu_user_pct[0:10]: {cpu_user_pct[0:10]}")
    #values = ewma(cpu_user_pct)
    #print(f"EWMA own values[0:10]: {values[0:10]}")
    print("_-------------------------------------------_")
    print(f"EWMA:{ewm_test['cpu_user_pct'].values[0:10]}")
    print("_-------------------------------------------_")