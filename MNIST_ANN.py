# simple_ann.py
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def relu(x): return np.maximum(0, x)
def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def train_ann(X, y_onehot, hidden=64, lr=0.1, epochs=100):
    # X: (n, d) normalized, y_onehot: (n, classes)
    n,d = X.shape
    c = y_onehot.shape[1]
    W1 = np.random.randn(d, hidden) * 0.01
    b1 = np.zeros(hidden)
    W2 = np.random.randn(hidden, c) * 0.01
    b2 = np.zeros(c)
    for _ in range(epochs):
        # forward
        h = relu(X @ W1 + b1)
        out = softmax(h @ W2 + b2)
        # loss grad (cross-entropy)
        grad_out = (out - y_onehot) / n
        dW2 = h.T @ grad_out
        db2 = grad_out.sum(axis=0)
        dh = grad_out @ W2.T
        dh[h <= 0] = 0
        dW1 = X.T @ dh
        db1 = dh.sum(axis=0)
        # update
        W2 -= lr * dW2; b2 -= lr * db2
        W1 -= lr * dW1; b1 -= lr * db1
    return (W1,b1,W2,b2)

def predict_ann(params, X):
    W1,b1,W2,b2 = params
    h = relu(X @ W1 + b1)
    out = softmax(h @ W2 + b2)
    return np.argmax(out, axis=1)

# NOTE: For MNIST you would load images into X (n,784) and create y_onehot.
# Load MNIST dataset

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize and reshape
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

# One-hot encode labels
y_train_onehot = to_categorical(y_train, 10)
y_test_onehot = to_categorical(y_test, 10)

# Train the model
params = train_ann(X_train, y_train_onehot, hidden=128, lr=0.1, epochs=50)

# Evaluate
predictions = predict_ann(params, X_test)
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy:.4f}")