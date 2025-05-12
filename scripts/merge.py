"""
final_merge.py

Combines 200 examples from human, GPT, Claude, and Mistral sources into a unified
dataset with columns: Prompt, text, label. Filters short/incomplete articles.
"""

import pandas as pd
import os

# Load and reformat human data
human_raw = pd.read_csv("data/news/human_200.csv").dropna(subset=["Title", "Articles"])
human_df = human_raw.rename(columns={"Title": "Prompt", "Articles": "text"})
human_df["label"] = "human"

# Load and rename AI sources
gpt_df = pd.read_csv("data/news/gpt_200.csv").rename(columns={"GPT4o_Response": "text"})
claude_df = pd.read_csv("data/news/claude_200.csv").rename(columns={"Claude_Response": "text"})
mistral_df = pd.read_csv("data/news/mistral_200.csv").rename(columns={"Mistral_Response": "text"})

# Use same prompt source from human_df
prompts = human_df["Prompt"].tolist()

gpt_df["Prompt"] = prompts
claude_df["Prompt"] = prompts
mistral_df["Prompt"] = prompts

gpt_df["label"] = "gpt"
claude_df["label"] = "claude"
mistral_df["label"] = "mistral"

# Ensure alignment
columns = ["Prompt", "text", "label"]
dfs = [human_df[columns], gpt_df[columns], claude_df[columns], mistral_df[columns]]

# Merge and clean
merged = pd.concat(dfs, ignore_index=True)
merged["text"] = merged["text"].str.strip()
# merged = merged[merged["text"].str.len() > 100]

# Save final dataset
os.makedirs("data/news", exist_ok=True)
merged.to_csv("data/news/merged_dataset.csv", index=False)

# Summary
print("Final merged dataset saved.")
print(merged["label"].value_counts())
print(f"Total examples: {len(merged)}")
