"""
news_visualize.py

Loads trained models and test data, runs evaluation + plots:
- Binary confusion matrix
- Multiclass confusion matrix
- Per-class metrics bar chart
- Confidence histogram
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import torch

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# === Load data ===
df_all = pd.read_csv("data/news/merged_dataset.csv")
df_all["binary_label"] = df_all["label"].apply(lambda x: 0 if x == "human" else 1)

# === Binary classifier eval ===
from sklearn.feature_extraction.text import TfidfVectorizer
clf = joblib.load("models/binary_classifier.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df_all['text'], df_all['binary_label'], test_size=0.2, random_state=42, stratify=df_all['binary_label'])

X_test_vec = vectorizer.transform(X_test)
y_pred = clf.predict(X_test_vec)

print("Binary Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))

cm_bin = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_bin, annot=True, fmt="d", xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Human vs AI) for News Articles")
plt.tight_layout()
plt.show()

# === Multiclass eval ===
df_ai = df_all[df_all['binary_label'] == 1][['text', 'label']].copy()
label2id = {'gpt': 0, 'claude': 1, 'mistral': 2}
id2label = {v: k for k, v in label2id.items()}
df_ai["label"] = df_ai["label"].map(label2id)

from datasets import Dataset
dataset = Dataset.from_pandas(df_ai)
split = dataset.train_test_split(test_size=0.2, seed=42)
test_ds = split["test"]

model = AutoModelForSequenceClassification.from_pretrained("models/multiclass_bert")
tokenizer = AutoTokenizer.from_pretrained("models/multiclass_bert")
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

texts = test_ds["text"]
inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = np.argmax(probs, axis=1)

y_true = test_ds["label"]
y_pred_named = [id2label[i] for i in preds]
y_true_named = [id2label[i] for i in y_true]

print("Multiclass Attribution Report:")
print(classification_report(y_true, preds, target_names=id2label.values()))

# Confusion matrix
labels = list(id2label.values())
cm = confusion_matrix(y_true_named, y_pred_named, labels=labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - AI Model Attribution in News Articles")
plt.tight_layout()
plt.show()

# Bar chart
labels = list(id2label.values())
precision, recall, f1, support = precision_recall_fscore_support(
    y_true_named, y_pred_named, labels=labels, zero_division=0)


import pandas as pd
metrics_df = pd.DataFrame({
    "Model": list(id2label.values()),
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1
})

metrics_df.set_index("Model").plot(kind="bar", figsize=(8, 6), rot=0, colormap="viridis")
plt.title("Precision, Recall, and F1-score by AI Model")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Confidence distribution
max_probs = probs.max(axis=1)
plt.figure(figsize=(6, 4))
sns.histplot(max_probs, bins=20, kde=True, color="skyblue")
plt.title("Prediction Confidence Distribution (Multiclass BERT)")
plt.xlabel("Max Probability")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
