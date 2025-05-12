"""
opinion_claude_generator.py

Generates Claude-3-Haiku articles from prompts in `opinion_prompts.csv`.
Saves results to `data/opinion/claude.csv`.
"""

import pandas as pd
import anthropic
import os
from dotenv import load_dotenv

load_dotenv("keys.env")
anthropic_api_key = os.getenv("anthropic_api_key")
client = anthropic.Anthropic(api_key=anthropic_api_key)

input_path = "data/opinion/opinion_prompts.csv"
output_path = "data/opinion/claude.csv"
os.makedirs("data/opinion", exist_ok=True)

df = pd.read_csv(input_path)
prompts = df["Title"].dropna().tolist()

results = []
for i, prompt in enumerate(prompts):
    print(f"[Claude] Generating article {i+1}/{len(prompts)}...")
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1200,
            temperature=0.7,
            system="You are a professional journalist at a major opinion outlet.",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Write a well-structured opinion article (~900 words) on the following topic:\n\n"
                        f"\"{prompt}\"\n\n"
                        "Maintain a formal, informed tone. Do not include any introductions or meta-text. Start directly with the article."
                    )
                }
            ]
        )
        article = response.content[0].text.strip()
    except Exception as e:
        article = f"[ERROR] {e}"
        print(f"❌ Error on prompt {i+1}: {e}")
    results.append({"Prompt": prompt, "Claude_Response": article})

pd.DataFrame(results).to_csv(output_path, index=False)
print(f"✅ Saved Claude articles to {output_path}")
