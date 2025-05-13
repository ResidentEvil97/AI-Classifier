"""
model_eval.py

Evaluates your trained binary and multiclass classifiers on the opinion dataset
(merged human + AI articles), to test cross-domain generalization.
"""

import pandas as pd
import numpy as np
import joblib
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ==== Load dataset ====
df = pd.read_csv("data/opinion/opinion_merged.csv")
df["text"] = df["text"].fillna("").str.strip()
df = df[df["text"].str.len() > 100]

# ==== Load binary classifier ====
clf = joblib.load("models/binary_classifier.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# ==== Predict Human vs AI ====
X_vec = vectorizer.transform(df["text"])
df["binary_pred"] = clf.predict(X_vec)
df["binary_true"] = df["label"].apply(lambda x: 0 if x == "human" else 1)

print("Binary Classification Report (Opinion Domain):")
print(classification_report(df["binary_true"], df["binary_pred"], target_names=["Human", "AI"]))

# ==== Confusion Matrix (Binary) ====
cm_bin = confusion_matrix(df["binary_true"], df["binary_pred"])
sns.heatmap(cm_bin, annot=True, fmt="d", xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix: Human vs AI (Opinion)")
plt.tight_layout()
plt.show()

# ==== Filter AI only for BERT attribution ====
df_ai = df[df["binary_true"] == 1].copy()

# ==== Load BERT classifier ====
model = AutoModelForSequenceClassification.from_pretrained("models/multiclass_bert")
tokenizer = AutoTokenizer.from_pretrained("models/multiclass_bert")
id2label = {0: "gpt", 1: "claude", 2: "mistral"}

# ==== Predict using BERT ====
inputs = tokenizer(df_ai["text"].tolist(), return_tensors="pt", padding=True, truncation=True, max_length=512)
with torch.no_grad():
    outputs = model(**inputs).logits
    probs = torch.softmax(outputs, dim=1).numpy()
    preds = np.argmax(probs, axis=1)

df_ai["multiclass_pred"] = preds
df_ai["multiclass_true"] = df_ai["label"].map({"gpt": 0, "claude": 1, "mistral": 2})

# ==== Evaluation ====
print("Multiclass Attribution Report (Opinion Domain):")
print(classification_report(df_ai["multiclass_true"], df_ai["multiclass_pred"], target_names=id2label.values()))

# ==== Confusion Matrix (Multiclass) ====
y_true_named = [id2label[i] for i in df_ai["multiclass_true"]]
y_pred_named = [id2label[i] for i in df_ai["multiclass_pred"]]
cm = confusion_matrix(y_true_named, y_pred_named, labels=id2label.values())

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=id2label.values(), yticklabels=id2label.values())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix: AI Attribution (Opinion)")
plt.tight_layout()
plt.show()

# ==== Confidence Histogram ====
max_probs = probs.max(axis=1)
plt.figure(figsize=(6, 4))
sns.histplot(max_probs, bins=20, kde=True, color="skyblue")
plt.title("Prediction Confidence (AI Attribution on Opinion Articles)")
plt.xlabel("Max Softmax Confidence")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
