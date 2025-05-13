"""
log_model_eval.py

Logs results of binary and multiclass classifiers on both:
1. Original news test set
2. Generalization to opinion articles

Saves all output to results/model_comparison.txt
"""

import os
import pandas as pd
import numpy as np
import joblib
import torch
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.makedirs("results", exist_ok=True)
output_file = "results/model_comparison.txt"

with open(output_file, "w") as f:

    # === Binary classifier on news data ===
    df_news = pd.read_csv("data/news/merged_dataset.csv")
    df_news["binary_label"] = df_news["label"].apply(lambda x: 0 if x == "human" else 1)

    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer

    clf = joblib.load("models/binary_classifier.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        df_news['text'], df_news['binary_label'], test_size=0.2, stratify=df_news['binary_label'], random_state=42)

    X_test_vec = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_vec)

    f.write("=== Binary Classifier on News ===\n")
    f.write(classification_report(y_test, y_pred, target_names=["Human", "AI"]))
    f.write("\n\n")

    # === Binary classifier on opinion data ===
    df_opinion = pd.read_csv("data/opinion/opinion_merged.csv")
    df_opinion["binary_label"] = df_opinion["label"].apply(lambda x: 0 if x == "human" else 1)

    X_opinion_vec = vectorizer.transform(df_opinion["text"])
    y_pred_opinion = clf.predict(X_opinion_vec)

    f.write("=== Binary Classifier on Opinion ===\n")
    f.write(classification_report(df_opinion["binary_label"], y_pred_opinion, target_names=["Human", "AI"]))
    f.write("\n\n")

    # === Multiclass BERT on news data ===
    f.write("=== Multiclass Classifier on News ===\n")
    from datasets import Dataset
    df_news_ai = df_news[df_news["binary_label"] == 1][["text", "label"]].copy()
    label2id = {'gpt': 0, 'claude': 1, 'mistral': 2}
    id2label = {v: k for k, v in label2id.items()}
    df_news_ai["label"] = df_news_ai["label"].map(label2id)

    dataset = Dataset.from_pandas(df_news_ai)
    split = dataset.train_test_split(test_size=0.2, seed=42)
    test_ds = split["test"]

    model = AutoModelForSequenceClassification.from_pretrained("models/multiclass_bert")
    tokenizer = AutoTokenizer.from_pretrained("models/multiclass_bert")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = tokenizer(test_ds["text"], return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    f.write(classification_report(test_ds["label"], preds, target_names=id2label.values()))
    f.write("\n\n")

    # === Multiclass BERT on opinion data ===
    f.write("=== Multiclass Classifier on Opinion ===\n")
    df_opinion_ai = df_opinion[df_opinion["label"] != "human"].copy()
    df_opinion_ai["label"] = df_opinion_ai["label"].map(label2id)

    inputs = tokenizer(df_opinion_ai["text"].tolist(), return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        preds_opinion = torch.argmax(logits, dim=1).cpu().numpy()

    f.write(classification_report(df_opinion_ai["label"], preds_opinion, target_names=id2label.values()))
    f.write("\n\n")

print(f"Results written to {output_file}")
