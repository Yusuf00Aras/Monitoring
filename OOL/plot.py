"""Plotting utilities for the OOL method.

Plots each feature with its frozen upper/lower limits and highlights
values that cross the limit.
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

from cleaning_utils import extract_all_features
from method import ool_init, ool_update, THRESHOLD, WARMUP


######
# Plot each feature with frozen limits and flagged anomalies
######

def plot_ool(data_path, threshold=THRESHOLD, warmup=WARMUP, max_features=None):
    """Plot each feature with frozen limits and flagged anomalies."""
    features, timestamps = extract_all_features(data_path)
    time_objs = [datetime.strptime(t[:19], "%Y-%m-%d %H:%M:%S") for t in timestamps]
    time = np.array(time_objs)

    names = list(features.keys())
    if max_features:
        names = names[:max_features]

    for name in names:
        values = np.array(features[name], dtype=float)
        state = ool_init(threshold, warmup)
        is_anomaly = np.zeros(len(values), dtype=bool)
        upper = np.full(len(values), np.nan)
        lower = np.full(len(values), np.nan)

        for i, v in enumerate(values):
            state, flag, _ = ool_update(state, v)
            is_anomaly[i] = flag
            if state['frozen_mean'] is not None and state['frozen_std'] is not None:
                upper[i] = state['frozen_mean'] + threshold * state['frozen_std']
                lower[i] = state['frozen_mean'] - threshold * state['frozen_std']

        plt.figure(figsize=(12, 5))
        plt.plot(time, values, color='steelblue', linewidth=1.2, label=name)
        plt.plot(time, upper, color='darkred', linestyle='--', linewidth=1, label=f'Upper limit (+{threshold}σ)')
        plt.plot(time, lower, color='darkred', linestyle='--', linewidth=1, label=f'Lower limit (−{threshold}σ)')
        plt.scatter(time[is_anomaly], values[is_anomaly], facecolors='none',
                    edgecolors='red', s=80, label='Anomaly')

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gcf().autofmt_xdate()
        plt.title(f'OOL — {name}')
        plt.xlabel('Time (HH:MM)')
        plt.ylabel(name)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else '../Test_Data/raw_data.csv'
    plot_ool(path)
