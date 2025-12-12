# knn_simple.py
import numpy as np
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
def knn_predict(X_train, y_train, X_test, k=3):
    preds = []
    for x in X_test:
        dists = np.sqrt(np.sum((X_train - x)**2, axis=1))
        idx = np.argsort(dists)[:k]
        majority = Counter(y_train[idx]).most_common(1)[0][0]
        preds.append(majority)
    return np.array(preds)
# Generate a synthetic dataset for classification (e.g., 2 features, 2 classes)
X_knn, y_knn = make_classification(n_samples=150, n_features=2, n_informative=2,
                                   n_redundant=0, n_clusters_per_class=1, random_state=42)
# Split the dataset into training and testing sets
X_knn_train, X_knn_test, y_knn_train, y_knn_test = train_test_split(X_knn, y_knn, test_size=0.2, random_state=42)
# Use the knn_predict function
k_value = 5 # You can change this value
y_knn_pred = knn_predict(X_knn_train, y_knn_train, X_knn_test, k=k_value)
# Calculate accuracy
accuracy_knn = accuracy_score(y_knn_test, y_knn_pred)
import matplotlib.pyplot as plt
# Plotting the KNN results (for 2 features)
plt.figure(figsize=(10, 6))
# Scatter plot of test data points
plt.scatter(X_knn_test[:, 0], X_knn_test[:, 1], c=y_knn_test, cmap='viridis', s=80, edgecolors='k', alpha=0.7, label='True Labels')
plt.scatter(X_knn_test[:, 0], X_knn_test[:, 1], c=y_knn_pred, cmap='plasma', marker='x', s=100, linewidths=2, label='Predictions (X)')
plt.title(f'k-Nearest Neighbors (k={k_value}) on Synthetic Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

'''
# Example:
# y_pred = knn_predict(X_train, y_train, X_test, k=5)

print("X_knn_train shape:", X_knn_train.shape)
print("y_knn_train shape:", y_knn_train.shape)
print("X_knn_test shape:", X_knn_test.shape)
print("y_knn_test shape:", y_knn_test.shape)
print(f"\nPredictions using KNN (k={k_value}):")

print("First 5 true y_knn_test values:")
print(y_knn_test[:5])
print("\nFirst 5 predicted y_knn_pred values:")
print(y_knn_pred[:5])
'''
print(f"\nAccuracy on the test set with k={k_value}: {accuracy_knn:.4f}")
