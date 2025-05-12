"""
model_trainer.py

Performs:
1. Binary classification (Human vs. AI) using TF-IDF + Logistic Regression.
2. Multiclass classification (GPT, Claude, Mistral) using fine-tuned BERT.

Outputs:
- Model performance reports
- Visualizations (confusion matrix, F1/precision/recall, confidence histogram)
- Trained models saved to disk
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from tqdm.auto import tqdm
import evaluate

# ====== LOAD DATA ======
df_all = pd.read_csv("data/news/merged_dataset.csv")
print(df_all['binary_label'].value_counts())

# ====== BINARY CLASSIFIER ======
X_train, X_test, y_train, y_test = train_test_split(
    df_all['text'], df_all['binary_label'],
    test_size=0.2, random_state=42, stratify=df_all['binary_label']
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = LogisticRegression(class_weight='balanced')
clf.fit(X_train_vec, y_train)

y_pred = clf.predict(X_test_vec)
print("Binary Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))

# Top features
feature_names = vectorizer.get_feature_names_out()
coefs = clf.coef_[0]
print("Top AI indicators:", [feature_names[i] for i in np.argsort(coefs)[-10:]])
print("Top Human indicators:", [feature_names[i] for i in np.argsort(coefs)[:10]])

# Save binary model + vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/binary_classifier.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print("Binary model and vectorizer saved.")

# ====== MULTICLASS BERT CLASSIFIER ======
df_ai = df_all[df_all['binary_label'] == 1][['text', 'label']].copy()
label2id = {'gpt': 0, 'claude': 1, 'mistral': 2}
id2label = {v: k for k, v in label2id.items()}
df_ai['label'] = df_ai['label'].map(label2id)

dataset = Dataset.from_pandas(df_ai)
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_ds = split_dataset['train']
test_ds = split_dataset['test']

model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)

def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

train_ds = train_ds.map(preprocess_function, batched=True)
test_ds = test_ds.map(preprocess_function, batched=True)

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Add TQDM training progress
class TQDMProgressBar(TrainerCallback):
    def __init__(self):
        self.pbar = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.pbar = tqdm(total=state.max_steps, desc="Training Progress")

    def on_step_end(self, args, state, control, **kwargs):
        self.pbar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        self.pbar.close()

training_args = TrainingArguments(
    output_dir="models/results",
    #evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    logging_dir="models/logs",
    report_to="none"  # disable W&B or Hugging Face Hub logging
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
    callbacks=[TQDMProgressBar()]
)

# Train
trainer.train()

# Evaluate
predictions = trainer.predict(test_ds)
pred_labels = np.argmax(predictions.predictions, axis=1)

print("Multiclass Classification Report:")
print(classification_report(test_ds["label"], pred_labels, target_names=id2label.values()))

# Save BERT model and tokenizer
model.save_pretrained("models/multiclass_bert")
tokenizer.save_pretrained("models/multiclass_bert")
print("BERT model and tokenizer saved.")

# ====== EVALUATION VISUALIZATIONS ======
# Confusion matrix for binary classifier
cm_bin = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_bin, annot=True, fmt="d", xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Human vs AI)")
plt.tight_layout()
plt.show()

y_true_named = [id2label[i] for i in test_ds["label"]]
y_pred_named = [id2label[i] for i in pred_labels]

# Confusion matrix
cm = confusion_matrix(y_true_named, y_pred_named, labels=id2label.values())
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=id2label.values(), yticklabels=id2label.values())
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - AI Model Attribution")
plt.tight_layout()
plt.show()

# Bar chart: Precision, Recall, F1
precision, recall, f1, support = precision_recall_fscore_support(
    y_true_named, y_pred_named, labels=id2label.values(), zero_division=0)

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

# Confidence Distribution
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
max_probs = probs.max(axis=1)

plt.figure(figsize=(6, 4))
sns.histplot(max_probs, bins=20, kde=True, color="skyblue")
plt.title("Prediction Confidence Distribution")
plt.xlabel("Max Probability (Confidence)")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
