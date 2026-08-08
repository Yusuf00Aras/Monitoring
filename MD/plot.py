import matplotlib.pyplot as plt
import numpy as np
from method import distances, test_data


def plot_distances(distances, X):
	mean = np.mean(X, axis=0)
	threshold = 1.5
	outliers = distances > threshold

	plt.figure(figsize=(8, 6))
	scatter = plt.scatter(X[:, 0], X[:, 1], c=distances, cmap='coolwarm', s=100, edgecolor='black')

	for i, d in enumerate(distances):
		plt.text(X[i, 0] + 0.5, X[i, 1], f'{d:.2f}', fontsize=9)

	plt.scatter(mean[0], mean[1], color='green', marker='X', s=200, label='Mean')
	plt.scatter(X[outliers, 0], X[outliers, 1], facecolors='none', edgecolors='red', s=200, label='Outlier')

	plt.title('Mahalanobis Distance Outlier Detection')
	plt.xlabel('Bidah')
	plt.ylabel('Salafi')
	plt.colorbar(scatter, label='Mahalanobis Distance')
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.show()


def main():
	plot_distances(distances, test_data)


if __name__ == "__main__":
	main()

