import csv
import matplotlib.pyplot as plt

filename = "oral_cancer_prediction_dataset.csv"

# --- Step 1: Load dataset ---
with open(filename, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    data = list(reader)
    columns = reader.fieldnames

print("Columns found in dataset:")
print(columns)
print("-" * 60)

# --- Step 2: Identify numeric + label columns ---
numeric_columns = []
label_col = None

for col in columns:
    numeric_count = 0
    for row in data[:30]:
        val = row.get(col, "").strip()
        try:
            float(val)
            numeric_count += 1
        except:
            pass
    if numeric_count > 10:
        numeric_columns.append(col)
    if any(x in col.lower() for x in ["cancer", "target", "label", "result", "diagnosis", "status"]):
        label_col = col

if not label_col:
    label_col = columns[-1]

print("Numeric columns:", numeric_columns)
print("Label column:", label_col)
print("-" * 60)

# --- Step 3: Extract clean data ---
features, labels = [], []

for row in data:
    vals = []
    for col in numeric_columns:
        val = row.get(col, "").strip()
        try:
            vals.append(float(val))
        except:
            continue
    if not vals:
        continue
    val_l = row.get(label_col, "").strip()
    if not val_l:
        continue
    l = 1 if val_l.lower() in ["1", "yes", "true", "positive", "cancer", "malignant"] else 0
    features.append(sum(vals)/len(vals))
    labels.append(l)

if not features:
    print(" No numeric data found.")
    exit()

# --- Step 4: Fast optimized thresholding ---
features, labels = zip(*sorted(zip(features, labels)))
n = len(features)

# Try 5 smart thresholds (20%, 40%, 50%, 60%, 80%)
percentiles = [0.2, 0.4, 0.5, 0.6, 0.8]
best_acc, best_threshold = 0, None

for p in percentiles:
    idx = int(p * n)
    threshold = features[idx]
    correct = sum(1 for i in range(n) if (features[i] > threshold) == (labels[i] == 1))
    acc = correct / n
    if acc > best_acc:
        best_acc, best_threshold = acc, threshold

print(f" Best Threshold: {round(best_threshold, 2)}")
print(f" Accuracy (fast rule-based): {round(best_acc * 100, 2)}%")
print("-" * 60)

# --- Step 5: Visualization ---
plt.hist(features, bins=10, color="skyblue", edgecolor="black")
plt.axvline(best_threshold, color='red', linestyle='--', label='Best Threshold')
plt.title("Feature Distribution with Decision Threshold")
plt.xlabel("Feature (avg. value)")
plt.ylabel("Count")
plt.legend()
plt.show()

plt.scatter(features, labels, c=["red" if l == 1 else "green" for l in labels])
plt.axvline(best_threshold, color='blue', linestyle='dotted', label='Decision Line')
plt.title("Feature vs Cancer Presence")
plt.xlabel("Feature (average value)")
plt.ylabel("Cancer (1=Yes, 0=No)")
plt.legend()
plt.show()
