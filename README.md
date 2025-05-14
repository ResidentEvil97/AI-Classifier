# AI Authorship Attribution

This repository contains the code and resources for training a machine learning model to distinguish between human-written and AI-generated text and attribute the text to specific AI models (GPT-4o, Claude-3-haiku-20240307, Mistral-7B-Instruct-v0.2).

## Overview

This project explores the ability of current machine learning techniques to detect AI-generated content and attribute it to specific models. The main goal is to investigate how well existing attribution methods perform with newer, more sophisticated large language models (LLMs).

## Dataset

- **Human-written Articles**: Collected from various online sources using a custom web scraper.
- **AI-generated Articles**: Generated using GPT-4o, Claude-3-haiku-20240307, and Mistral-7B-Instruct-v0.2.
- **Genres**: News and Opinion articles were used for training and evaluation.

## Structure

- `data/`: Contains scripts for scraping data and generating AI content.
- `model_trainer.py`: Contains code to train binary and multiclass classifiers.
- `model_eval.py`: Includes functions for evaluating model performance and generating results.
- `opinion_scraper.py`: A script to scrape opinion articles from the web.
- `opinion_merge.py`: Merges scraped opinion data into a single dataset.
- `news_ai_generator.py`: Generates AI articles based on news topics.
- `README.md`: This file.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-authorship-attribution.git
   cd ai-authorship-attribution
