# linear_svm_sgd.py
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

def train_linear_svm(X, y, lr=0.01, epochs=200, C=1.0):
    # y should be -1 or +1
    Xb = np.hstack([np.ones((X.shape[0],1)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(epochs):
        # stochastic style: iterate rows
        for i in range(Xb.shape[0]):
            xi = Xb[i]
            yi = y[i]
            margin = yi * (w @ xi)
            if margin < 1:
                w = w + lr * (yi * xi - 2*(1/epochs)*w)
            else:
                w = w - lr * (2*(1/epochs) * w)
    return w

def predict_svm(w, X):
    Xb = np.hstack([np.ones((X.shape[0],1)), X])
    return np.where(Xb @ w >= 0, 1, -1)

# Example:
# convert spam labels to +1 (spam) and -1 (not spam), vectorize email text (simple bag-of-words).
# Example with a simple text dataset
emails = [
    "buy now limited offer",
    "hello how are you",
    "click here to win prize",
    "meeting at 3pm today",
    "free money click link",
    "project update attached"
]
labels = np.array([1, -1, 1, -1, 1, -1])  # 1 for spam, -1 for not spam

# Vectorize emails using bag-of-words
vectorizer = CountVectorizer(lowercase=True, stop_words='english')
X = vectorizer.fit_transform(emails).toarray()

# Train and predict
w = train_linear_svm(X, labels, lr=0.01, epochs=100, C=1.0)
predictions = predict_svm(w, X)
print("Predictions:", predictions)
print("Accuracy:", np.mean(predictions == labels))

