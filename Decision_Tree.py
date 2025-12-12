# decision_stump.py
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
import matplotlib.patches as patches

def best_stump(X, y):
    # returns (feature_index, threshold, left_label, right_label)
    n, d = X.shape
    best = None
    best_err = n+1
    for j in range(d):
        vals = np.unique(X[:,j])
        thresholds = (vals[:-1] + vals[1:]) / 2 if len(vals)>1 else vals
        for t in thresholds:
            left = y[X[:,j] <= t]
            right = y[X[:,j] > t]
            l_label = np.round(np.mean(left)) if len(left)>0 else 0
            r_label = np.round(np.mean(right)) if len(right)>0 else 0
            err = np.sum(left != l_label) + np.sum(right != r_label)
            if err < best_err:
                best_err = err
                best = (j, t, int(l_label), int(r_label))
    return best

def predict_stump(stump, X):
    j,t,l,r = stump
    return np.where(X[:,j] <= t, l, r)

# Example:
# stump = best_stump(X_train, y_train)
# y_pred = predict_stump(stump, X_test)
# Generate a sample dataset
X, y = make_classification(n_samples=200, n_features=5, n_informative=3, 
                           n_redundant=0, random_state=42)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the decision stump
stump = best_stump(X_train, y_train)
print(f"Best stump: feature={stump[0]}, threshold={stump[1]:.4f}")

# Make predictions
y_pred = predict_stump(stump, X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")


j, t, l, r = stump   # from your code

plt.figure(figsize=(6,4))
plt.axis('off')

# Layer 1 (root)
plt.text(0.5, 0.92, f"Feature {j} ≤ {t:.3f} ?", ha='center', fontsize=12,
         bbox=dict(boxstyle="round", fc="#d0e6ff"))

# Layer 2 (children)
plt.text(0.25, 0.68, f"Predict {l}", ha='center', fontsize=11, bbox=dict(boxstyle="round", fc="#ffe6cc"))
plt.text(0.75, 0.68, f"Predict {r}", ha='center', fontsize=11, bbox=dict(boxstyle="round", fc="#ffe6cc"))
plt.annotate("", xy=(0.25,0.72), xytext=(0.5,0.88), arrowprops=dict(arrowstyle="->"))
plt.annotate("", xy=(0.75,0.72), xytext=(0.5,0.88), arrowprops=dict(arrowstyle="->"))
plt.text(0.37,0.82,"True",fontsize=9); plt.text(0.63,0.82,"False",fontsize=9)

# Layer 3 (leaves: two under each child)
xs = [0.125, 0.375, 0.625, 0.875]
for i,x in enumerate(xs):
    plt.text(x, 0.42, "Leaf", ha='center', fontsize=9, bbox=dict(boxstyle="round", fc="#f0f8ff"))
# arrows from left child to its two leaves
plt.annotate("", xy=(xs[0],0.46), xytext=(0.25,0.64), arrowprops=dict(arrowstyle="->"))
plt.annotate("", xy=(xs[1],0.46), xytext=(0.25,0.64), arrowprops=dict(arrowstyle="->"))
# arrows from right child to its two leaves
plt.annotate("", xy=(xs[2],0.46), xytext=(0.75,0.64), arrowprops=dict(arrowstyle="->"))
plt.annotate("", xy=(xs[3],0.46), xytext=(0.75,0.64), arrowprops=dict(arrowstyle="->"))

plt.title("Decision Stump — 3-Layer Illustration")
plt.tight_layout()
plt.show()