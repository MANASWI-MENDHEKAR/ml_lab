# pca_svd.py
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def pca(X, n_components=2):
    # X: (n, d) -- center it
    Xc = X - X.mean(axis=0)
    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]
    transformed = Xc @ components.T
    return transformed, components

# Example:
# X2, comps = pca(X, n_components=2)
# Load a sample dataset

# Load the iris dataset
data = load_iris()
X = data.data
y = data.target

# Apply PCA
X_transformed, components = pca(X, n_components=2)

# Print the results
print("Transformed Data:\n", X_transformed)
print("Principal Components:\n", components)
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y, cmap='viridis')
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("PCA (SVD) - Iris Dataset")
plt.colorbar(label="Class")
plt.tight_layout()
plt.show()