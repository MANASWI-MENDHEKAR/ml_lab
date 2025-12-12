# logistic_regression.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def fit_logistic(X, y, lr=0.1, epochs=200):
    # X: (n, d), y: {0,1}
    Xb = np.hstack([np.ones((X.shape[0],1)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(epochs):
        preds = sigmoid(Xb @ w)
        grad = Xb.T @ (preds - y) / Xb.shape[0]
        w -= lr * grad
    return w

def predict_logistic(w, X, threshold=0.5):
    Xb = np.hstack([np.ones((X.shape[0],1)), X])
    return (sigmoid(Xb @ w) >= threshold).astype(int)

# Generate a synthetic dataset for binary classification
X_log, y_log = make_classification(n_samples=100, n_features=2, n_informative=2, 
                                   n_redundant=0, n_clusters_per_class=1, random_state=42)

# Split the dataset into training and testing sets
X_log_train, X_log_test, y_log_train, y_log_test = train_test_split(X_log, y_log, test_size=0.2, random_state=42)

# Use the fit_logistic function to train the model
w = fit_logistic(X_log_train, y_log_train)

# Use the predict_logistic function to make predictions
y_log_pred = predict_logistic(w, X_log_test)

# Calculate accuracy
accuracy = accuracy_score(y_log_test, y_log_pred)
# Plot the decision boundary (for 2 features)
plt.figure(figsize=(10, 6))

# Plot test data points
plt.scatter(X_log_test[:, 0], X_log_test[:, 1], c=y_log_test, cmap='viridis', edgecolors='k', s=80, label='True labels')

# Create a meshgrid to plot the decision boundary
x_min, x_max = X_log[:, 0].min() - 1, X_log[:, 0].max() + 1
y_min, y_max = X_log[:, 1].min() - 1, X_log[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                       np.arange(y_min, y_max, 0.1))

Z = predict_logistic(w, np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
'''
# Example:
# w = fit_logistic(X_train, y_train)
# y_pred = predict_logistic(w, X_test)
print("X_log_train shape:", X_log_train.shape)
print("y_log_train shape:", y_log_train.shape)
print("X_log_test shape:", X_log_test.shape)
print("y_log_test shape:", y_log_test.shape)
print("Learned parameters (w):")
print(w)

print("\nFirst 5 true y_log_test values:")
print(y_log_test[:5])

print("\nFirst 5 predicted y_log_pred values:")
print(y_log_pred[:5])
'''
print(f"\nAccuracy on the test set: {accuracy:.4f}")
