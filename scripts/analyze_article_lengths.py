import pandas as pd
import matplotlib.pyplot as plt
import os

# Load human article dataset
df = pd.read_csv("data/opinion/opinion_human.csv")

# Compute word counts
df['word_count'] = df['text'].str.split().apply(len)

# Summary stats
print("Human Article Word Count Stats:")
print(df['word_count'].describe())

# Optional: histogram
plt.figure(figsize=(6, 4))
plt.hist(df['word_count'], bins=15, color="skyblue", edgecolor="black")
plt.title("Distribution of Human Article Word Counts")
plt.xlabel("Words per Article")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("data/opinion/opinion_word_hist.png")
plt.show()
