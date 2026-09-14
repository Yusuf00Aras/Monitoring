import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
from cleaning_utils import extract_important_features
from method import mahalanobis_distances


######
# Plot Mahalanobis distances over time with outlier threshold
######

def plot_distances(distances, time_strings):
    distances = np.array(distances)
    time_objs = [datetime.strptime(t[:19], "%Y-%m-%d %H:%M:%S") for t in time_strings]
    time = np.array(time_objs)

    mean = np.mean(distances)
    threshold = mean + 3 * np.std(distances)
    outliers = distances > threshold

    plt.figure(figsize=(12, 6))
    plt.plot(time, distances, color='steelblue', linewidth=1.5, label='Mahalanobis distance')
    plt.scatter(time[outliers], distances[outliers], facecolors='none',
                edgecolors='red', s=100, label='Outlier')
    plt.axhline(threshold, color='darkred', linestyle='--', linewidth=1.5, label='Threshold')

    for i, d in enumerate(distances):
        if outliers[i]:
            plt.text(time[i], d + (max(distances) * 0.02), f'{d:.2f}',
                     fontsize=8, ha='center')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate()
    plt.title('Mahalanobis Distance Over Time')
    plt.xlabel('Time (HH:MM)')
    plt.ylabel('Mahalanobis Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    features, time_stamps = extract_important_features('./Test_Data/raw_data.csv')
    distances = mahalanobis_distances(features)
    plot_distances(distances, time_stamps)