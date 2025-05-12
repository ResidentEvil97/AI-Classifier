"""
opinion_merge.py

Combines aligned opinion articles from all sources into a unified dataset
with columns: Prompt, Text, Label
"""

import pandas as pd
import os

# Load and standardize all datasets
human_df = pd.read_csv("data/opinion/human.csv").rename(columns={"Human_Response": "text"})
gpt_df = pd.read_csv("data/opinion/gpt.csv").rename(columns={"GPT4o_Response": "text"})
claude_df = pd.read_csv("data/opinion/claude.csv").rename(columns={"Claude_Response": "text"})
mistral_df = pd.read_csv("data/opinion/mistral.csv").rename(columns={"Mistral_Response": "text"})

# Add labels
human_df["label"] = "human"
gpt_df["label"] = "gpt"
claude_df["label"] = "claude"
mistral_df["label"] = "mistral"

# Ensure alignment by matching on prompts
columns = ["Prompt", "text", "label"]
dfs = [human_df[columns], gpt_df[columns], claude_df[columns], mistral_df[columns]]

# Merge
merged = pd.concat(dfs, ignore_index=True)
merged["text"] = merged["text"].str.strip()
merged = merged[merged["text"].str.len() > 100]

# Save
os.makedirs("data/opinion", exist_ok=True)
merged.to_csv("data/opinion/opinion_merged.csv", index=False)
print(f"Saved merged dataset with {len(merged)} examples to data/opinion/opinion_merged.csv")
