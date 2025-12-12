# breast_cancer_simple.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset (all numeric, last col = label)
data = np.loadtxt("breast_cancer.csv", delimiter=",")
X, y = data[:, :-1], data[:, -1].astype(int)

# Split data
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model
clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)

# Predict probabilities & final labels
proba = clf.predict_proba(X_te)[:, 1]
pred = (proba >= 0.5).astype(int)

# Accuracy
acc = (pred == y_te).mean()
print("Accuracy:", round(acc, 4))

# Simple Plot
plt.scatter(range(len(proba)), proba, c=y_te, cmap='coolwarm')
plt.axhline(0.5, color='black', linestyle='--')
plt.title("Breast Cancer Prediction")
plt.xlabel("Sample Index")
plt.ylabel("Cancer Probability")
plt.show()
