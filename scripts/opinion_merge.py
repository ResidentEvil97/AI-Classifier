"""
opinion_merge.py

Combines human and AI-generated opinion articles into a single labeled dataset.
Saves result to `data/opinion/opinion_merged.csv`.
"""

import pandas as pd
import os

# Load all sources
human_df = pd.read_csv("data/opinion/opinion_human.csv")[["text"]].dropna().copy()
gpt_df = pd.read_csv("data/opinion/gpt.csv")[["GPT4o_Response"]].rename(columns={"GPT4o_Response": "text"})
claude_df = pd.read_csv("data/opinion/claude.csv")[["Claude_Response"]].rename(columns={"Claude_Response": "text"})
mistral_df = pd.read_csv("data/opinion/mistral.csv")[["Mistral_Response"]].rename(columns={"Mistral_Response": "text"})

# Add labels
human_df["label"] = "human"
gpt_df["label"] = "gpt"
claude_df["label"] = "claude"
mistral_df["label"] = "mistral"

# Combine
merged = pd.concat([human_df, gpt_df, claude_df, mistral_df], ignore_index=True)
merged["text"] = merged["text"].str.strip()
merged = merged[merged["text"].str.len() > 100]  # basic filtering

# Save
os.makedirs("data/opinion", exist_ok=True)
merged.to_csv("data/opinion/opinion_merged.csv", index=False)
print(f"Saved merged dataset with {len(merged)} examples to data/opinion/opinion_merged.csv")
