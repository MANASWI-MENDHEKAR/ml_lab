import csv
import matplotlib.pyplot as plt

# --- Step 1: Load Dataset ---
filename = "breast-cancer.csv"  # Make sure your CSV is in the same folder

radius_values = []
texture_values = []
labels = []

with open(filename, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            # Extract two features and label
            radius = float(row['radius_mean'])
            texture = float(row['texture_mean'])
            label = row['diagnosis']
            
            # Store values
            radius_values.append(radius)
            texture_values.append(texture)
            labels.append(1 if label == 'M' else 0)  # 1 = Malignant, 0 = Benign
        except:
            continue

# --- Step 2: Compute Feature Thresholds ---
radius_threshold = sum(radius_values) / len(radius_values)
texture_threshold = sum(texture_values) / len(texture_values)

# --- Step 3: Make Predictions ---
predictions = []
for r, t in zip(radius_values, texture_values):
    # Simple rule: if both features exceed their thresholds, predict cancer
    if r > radius_threshold and t > texture_threshold:
        predictions.append(1)
    else:
        predictions.append(0)

# --- Step 4: Calculate Accuracy ---
correct = sum(1 for i in range(len(predictions)) if predictions[i] == labels[i])
accuracy = correct / len(predictions) * 100

# --- Step 5: Display Results ---
print("Total samples:", len(predictions))
print(f"Thresholds -> radius_mean: {radius_threshold:.2f}, texture_mean: {texture_threshold:.2f}")
print(f"Accuracy (2-feature rule-based): {accuracy:.2f}%")
print("-" * 60)

# --- Step 6: Visualizations ---

# Scatter plot (radius vs texture)
plt.figure(figsize=(8, 6))
for i in range(len(radius_values)):
    color = 'red' if labels[i] == 1 else 'blue'
    plt.scatter(radius_values[i], texture_values[i], color=color)

plt.axvline(radius_threshold, color='green', linestyle='--', label='radius threshold')
plt.axhline(texture_threshold, color='orange', linestyle='--', label='texture threshold')
plt.xlabel('radius_mean')
plt.ylabel('texture_mean')
plt.title('Breast Cancer Detection (2-Feature Rule-Based Model)')
plt.legend()
plt.show()

# Bar chart (Cancer vs Non-Cancer)
cancer_count = labels.count(1)
non_cancer_count = labels.count(0)
plt.bar(['Benign', 'Malignant'], [non_cancer_count, cancer_count], color=['green', 'yellow'])
plt.title('Cancer Type Distribution')
plt.ylabel('Number of Samples')
plt.show()

