# # oral_cancer_minimal_with_plot.py
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, roc_auc_score

# # Load CSV (all numeric, last column = label 0/1)
# data = np.loadtxt("oral_cancer.csv", delimiter=",")
# X, y = data[:, :-1], data[:, -1].astype(int)

# # Train-test split
# Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Train model
# clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)

# # Predictions
# proba = clf.predict_proba(Xte)[:, 1]
# pred = (proba >= 0.5).astype(int)

# print("Accuracy:", round(accuracy_score(yte, pred), 4))
# print("ROC AUC:", round(roc_auc_score(yte, proba), 4))

# # ----- Plot -----
# plt.figure(figsize=(8,4))
# plt.scatter(range(len(proba)), proba, c=yte, cmap='coolwarm', s=50)
# plt.axhline(0.5, color='black', linestyle='--', label='Decision Boundary (0.5)')
# plt.title("Oral Cancer Prediction (Probability Plot)")
# plt.xlabel("Sample Index")
# plt.ylabel("Predicted Cancer Probability")
# plt.colorbar(label="True Label (0 = Healthy, 1 = Cancer)")
# plt.legend()
# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

def load_csv_safely(path):
    # Load using genfromtxt (handles headers + text)
    raw = np.genfromtxt(path, delimiter=",", dtype=str)

    # Remove header
    data = raw[1:]

    # Convert everything except non-numeric columns
    numeric_data = []
    for col in data.T:  # iterate column-wise
        try:
            numeric_data.append(col.astype(float))
        except ValueError:
            # Column is non-numeric → skip it (ID or text)
            continue

    numeric_data = np.array(numeric_data).T  # transpose back to rows
    return numeric_data

# ---------------- MAIN SCRIPT ----------------

def main():
    # Load dataset safely
    data = load_csv_safely("oral_cancer_prediction_dataset.csv")

    # Last column = label
    X = data[:, :-1]
    y = data[:, -1].astype(int)

    # Basic rule-based prediction (no ML)
    threshold = X[:, 0].mean()
    y_pred = (X[:, 0] > threshold).astype(int)

    # Accuracy
    acc = (y_pred == y).mean()
    print("Accuracy:", round(acc, 4))

    # Simple graph
    plt.scatter(range(len(y_pred)), y_pred, c=y, cmap='coolwarm')
    plt.axhline(0.5, color='black', linestyle='--')
    plt.title("Oral Cancer Prediction (Rule-Based)")
    plt.xlabel("Sample Index")
    plt.ylabel("Predicted Label")
    plt.show()

if __name__ == "__main__":
    main()
