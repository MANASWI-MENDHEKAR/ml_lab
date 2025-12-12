# kmeans_simple.py
import numpy as np
import random
from sklearn.datasets import make_blobs

def kmeans(X, k=3, epochs=50):
    n,d = X.shape
    centers = X[np.random.choice(n, k, replace=False)]
    for _ in range(epochs):
        dists = np.linalg.norm(X[:,None,:] - centers[None,:,:], axis=2)  # (n,k)
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([X[labels==i].mean(axis=0) if np.any(labels==i) else centers[i] for i in range(k)])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return centers, labels

# Example:
# centers, labels = kmeans(X, k=4)
# Generate a sample dataset
import matplotlib.pyplot as plt

# Create a dataset with 300 samples, 2 features, and 4 centers
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Run the kmeans algorithm
centers, labels = kmeans(X, k=4)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()