"""
opinion_mistral_generator.py

Generates Mistral responses from prompts in `opinion_prompts.csv`.
Saves results to `data/opinion/mistral.csv`.
"""

import pandas as pd
import requests
import os
import time
from dotenv import load_dotenv

load_dotenv("keys.env")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

input_path = "data/opinion/opinion_prompts.csv"
output_path = "data/opinion/mistral.csv"
os.makedirs("data/opinion", exist_ok=True)

df = pd.read_csv(input_path)
prompts = df["Title"].dropna().tolist()

model = "mistralai/Mistral-7B-Instruct-v0.2"
url = "https://api.together.xyz/v1/chat/completions"
headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}

def format_prompt(prompt):
    return (
        "You are a professional journalist for a reputable opinion outlet.\n\n"
        "Write a 900-word opinion article based on the following topic:\n\n"
        f"\"{prompt}\"\n\n"
        "Be direct and factual. Avoid fluff. Do not explain your actions or say you are an AI."
    )

results = []
for i, prompt in enumerate(prompts):
    print(f"[Mistral] Generating article {i+1}/{len(prompts)}...")
    try:
        response = requests.post(
            url,
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a neutral, direct opinion writer."},
                    {"role": "user", "content": format_prompt(prompt)}
                ],
                "temperature": 0.7,
                "max_tokens": 1200
            },
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            article = data['choices'][0]['message']['content'].strip()
        else:
            article = f"[HTTP {response.status_code}] {response.text}"
            print(f"Error: {article}")
    except Exception as e:
        article = f"[ERROR] {e}"
        print(f" Exception on prompt {i+1}: {e}")
    
    results.append({"Prompt": prompt, "Mistral_Response": article})
    time.sleep(1)

pd.DataFrame(results).to_csv(output_path, index=False)
print(f"Saved Mistral articles to {output_path}")
