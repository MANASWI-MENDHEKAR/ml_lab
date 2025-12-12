import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Data (same as your original snippet)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

def fit_linear_regression(X, y):
    Xb = np.hstack([np.ones((X.shape[0],1)), X])
    theta = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y
    return theta

def predict_linear(theta, X):
    Xb = np.hstack([np.ones((X.shape[0],1)), X])
    return Xb @ theta

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

theta = fit_linear_regression(X_train, y_train.flatten())
y_pred = predict_linear(theta, X_test).flatten()
y_test = y_test.flatten()


# Binarize using median threshold (change threshold as needed)
threshold = np.median(y_test)
y_test_bin = (y_test >= threshold).astype(int)
y_pred_bin = (y_pred >= threshold).astype(int)

# Classification metrics
accuracy = accuracy_score(y_test_bin, y_pred_bin)
precision = precision_score(y_test_bin, y_pred_bin, zero_division=0)
recall = recall_score(y_test_bin, y_pred_bin, zero_division=0)
f1 = f1_score(y_test_bin, y_pred_bin, zero_division=0)
cm = confusion_matrix(y_test_bin, y_pred_bin)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print("Confusion matrix:\n", cm)
print("\nClassification report:\n", classification_report(y_test_bin, y_pred_bin, zero_division=0))

# Plot
plt.figure(figsize=(10,6))
plt.scatter(X_test, y_test, label='True values')
plt.scatter(X_test, y_pred, label='Predictions', alpha=0.7)
order = np.argsort(X_test.flatten())
plt.plot(X_test.flatten()[order], y_pred[order])
plt.xlabel('X (Feature)')
plt.ylabel('y (Target)')
plt.title('Linear Regression: True vs. Predicted Values')
plt.legend()
plt.grid(True)
plt.show()
