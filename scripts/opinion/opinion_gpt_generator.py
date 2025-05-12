"""
opinion_gpt_generator.py

Generates GPT-4o longform opinion-style articles from prompts and saves them to CSV.
Targets ~900 words using max_tokens=1200.
"""

import os
import openai
import pandas as pd
from dotenv import load_dotenv

# Load API key
load_dotenv("keys.env")
openai.api_key = os.getenv("openai.api_key")

# File paths
input_path = "data/opinion/opinion_prompts.csv"
output_path = "data/opinion/gpt.csv"
os.makedirs("data/opinion", exist_ok=True)

# Load prompts
df = pd.read_csv(input_path)
prompts = df['Title'].dropna().tolist()

# Generation loop
results = []
for i, prompt in enumerate(prompts):
    print(f"[GPT-4o] Generating article {i+1}/{len(prompts)}...")
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are a professional journalist for a major opinion outlet.\n\n"
                        "Write a thought-provoking, well-structured article (~900 words) on the topic below, matching the tone of publications like The Atlantic, Vox, or Scientific American.\n\n"
                        f"Title: \"{prompt}\"\n\n"
                        "Do not include any disclaimers or instructions — just begin with the article."
                    )
                }
            ],
            temperature=0.7,
            max_tokens=1200
        )
        article = response.choices[0].message.content.strip()
    except Exception as e:
        article = f"[ERROR] {e}"
        print(f"Error on prompt {i+1}: {e}")
    
    results.append({"Prompt": prompt, "GPT4o_Response": article})

# Save to CSV
pd.DataFrame(results).to_csv(output_path, index=False)
print(f"Saved GPT-4o outputs to {output_path}")
