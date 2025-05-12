"""
news_ai_generator.py

This script regenerates longform news-style articles from AI models (GPT-4o, Claude, Mistral)
based on a shared set of prompts. It is designed for local use as part of the "Which AI Said That?" 
authorship attribution project.

Each model receives the same prompt and produces a ~600–700 word article in journalistic tone. 
The outputs are saved to CSV files for downstream use in classification tasks.

Note: This script was created for reproducibility and demo purposes.
The original dataset used in this project was generated earlier using a similar version 
of this script in Google Colab and may differ slightly in output format or API settings.
"""


import openai
import pandas as pd
import os
import requests
import time
import anthropic
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv("keys.env")

# ========== CONFIG ==========
openai.api_key = os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

PROMPT_PATH = "data/news/human_sampled_130.csv"
OUTPUT_DIR = "data/news"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== Shared Prompt Template ==========
def make_prompt(title):
    return (
        "You are a professional journalist for a major news outlet.\n\n"
        "Write a news article (600–700 words) based on the title below, matching typical online news structure.\n"
        f"Title: \"{title}\"\n\n"
        "Do not include any introductions, explanations, or disclaimers. Start directly with the article."
    )

# ========== Load Prompts ==========
df = pd.read_csv(PROMPT_PATH)
prompts = df['Title'].dropna().tolist()

# ========== GPT-4o Generation ==========
def generate_gpt(prompts):
    results = []
    for i, prompt in enumerate(prompts):
        print(f"[GPT-4o] Generating article {i+1}/{len(prompts)}...")
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": make_prompt(prompt)
                }],
                temperature=0.7,
                max_tokens=950
            )
            reply = response.choices[0].message.content
        except Exception as e:
            reply = f"[ERROR] {e}"
        results.append({'Prompt': prompt, 'GPT4o_Response': reply})
    pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/gpt_extra.csv", index=False)

# ========== Claude Generation ==========
def generate_claude(prompts):
    client = anthropic.Anthropic(api_key=anthropic_api_key)
    results = []
    for i, prompt in enumerate(prompts):
        print(f"[Claude] Generating article {i+1}/{len(prompts)}...")
        try:
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=950,
                temperature=0.7,
                system="You are a helpful AI assistant.",
                messages=[{
                    "role": "user",
                    "content": make_prompt(prompt)
                }]
            )
            reply = response.content[0].text
        except Exception as e:
            reply = f"[ERROR] {e}"
        results.append({"Prompt": prompt, "Claude_Response": reply})
    pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/claude_extra.csv", index=False)

# ========== Mistral Generation ==========
def generate_mistral(prompts):
    model = "mistralai/Mistral-7B-Instruct-v0.2"
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}

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
                        {"role": "system", "content": "You are a helpful and neutral news writer."},
                        {"role": "user", "content": make_prompt(prompt)}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 950
                },
                timeout=60
            )
            if response.status_code == 200:
                data = response.json()
                reply = data['choices'][0]['message']['content']
            else:
                reply = f"[HTTP {response.status_code}] {response.text}"
        except Exception as e:
            reply = f"[ERROR] {e}"
        results.append({"Prompt": prompt, "Mistral_Response": reply})
        time.sleep(1)
    pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/mistral_extra.csv", index=False)

# ========== Run All ==========
if __name__ == "__main__":
    generate_gpt(prompts)
    generate_claude(prompts)
    generate_mistral(prompts)
    print("All generations complete and saved.")
