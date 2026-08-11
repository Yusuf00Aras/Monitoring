import matplotlib.pyplot as plt
import numpy as np
from method import features, mahalanobis_distances


def plot_distances(distances):
	time = np.arange(len(distances))
	mean = np.mean(distances)
	threshold = mean + 3 * np.std(distances)
	outliers = distances > threshold

	plt.figure(figsize=(10, 6))
	plt.plot(time, distances, color='steelblue', linewidth=1.5, label='Mahalanobis distance')
	plt.scatter(time[outliers], distances[outliers], facecolors='none', edgecolors='red', s=100, label='Outlier')
	plt.axhline(threshold, color='darkred', linestyle='--', linewidth=1.5, label='Threshold')

	for i, d in enumerate(distances):
		if outliers[i]:
			plt.text(time[i], d + 0.05, f'{d:.2f}', fontsize=8, ha='center')

	plt.title('Mahalanobis Distance Over Time')
	plt.xlabel('Time / Minute')
	plt.ylabel('Mahalanobis Distance')
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.show()



if __name__ == "__main__":
	distances = mahalanobis_distances(features)
	plot_distances(distances)


