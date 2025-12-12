# naive_bayes_text.py
import numpy as np
from collections import defaultdict, Counter
import re
import matplotlib.pyplot as plt

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def train_nb(docs, labels):
    # docs: list of strings; labels: list of 0/1
    vocab = set()
    word_counts = {0:Counter(), 1:Counter()}
    class_counts = Counter(labels)
    for doc,lab in zip(docs, labels):
        tokens = tokenize(doc)
        vocab.update(tokens)
        word_counts[lab].update(tokens)
    V = len(vocab)
    # compute log priors and log likelihoods with Laplace smoothing
    total_words = {c: sum(word_counts[c].values()) for c in [0,1]}
    def score(doc, c):
        tokens = tokenize(doc)
        logp = np.log(class_counts[c] / sum(class_counts.values()))
        for t in tokens:
            count = word_counts[c][t]
            logp += np.log((count + 1) / (total_words[c] + V))
        return logp
    return score

# Example usage:
# score = train_nb(train_docs, train_labels)
# pred = 1 if score("great product") > score("great product",0) else 0
# (for convenience you can wrap score into a predict function)
# Sample dataset
train_docs = [
    "This product is amazing and works great",
    "Excellent quality, highly recommended",
    "Best purchase ever, very satisfied",
    "Terrible quality, waste of money",
    "Broke after one day, very disappointed",
    "Awful experience, do not buy"
]
train_labels = [1, 1, 1, 0, 0, 0]

# Train the model
score = train_nb(train_docs, train_labels)

# Prediction function
def predict(doc):
    score0 = score(doc, 0)
    score1 = score(doc, 1)
    return 1 if score1 > score0 else 0, score0 , score1

# Test predictions
test_docs = ["Great product", "Very bad", "Excellent item"]
preds = []
scores0 = []
scores1 = []
for doc in test_docs:
    pred, s0, s1 = predict(doc)
    preds.append(pred)
    scores0.append(s0)
    scores1.append(s1)
    print(f"'{doc}' -> Class {pred}")

x = np.arange(len(test_docs))

plt.bar(x - 0.15, scores0, width=0.3, label="Class 0 Score")
plt.bar(x + 0.15, scores1, width=0.3, label="Class 1 Score")
plt.xticks(x, test_docs, rotation=20)
plt.ylabel("Log Score")
plt.title("Naive Bayes Class Scores")
plt.legend()
plt.tight_layout()
plt.show()