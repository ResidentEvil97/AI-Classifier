# AI Authorship Attribution

This repository contains the code and resources for training machine learning models to classify text as human-written or AI-generated and to attribute AI-generated text to specific models (GPT-4o, Claude-3-haiku-20240307, Mistral-7B-Instruct-v0.2).

## Overview

The tool uses trained classifiers to:

- Classify text as human-written or AI-generated.
- Attribute AI-generated text to one of three models: GPT-4o, Claude, or Mistral.
- Models were trained on news and opinion articles.

## Dataset

- **Human-written Articles**: Collected from various sources.
- **AI-generated Articles**: Generated using GPT-4o, Claude-3-haiku-20240307, and Mistral-7B-Instruct-v0.2.
- **Genres**: News articles were used for training and evaluation, and opinion articles were used for testing only.

## Structure
```
AI-Classifer/
├── app/
│   ├── app.py                  # Flask app for serving the web interface
│   ├── static/
│   │   └── style.css           # Styles for the web app
│   ├── templates/
│   │   ├── index.html          # Analysis page where text is submitted
│   │   └── landing_page.html   # Landing page template
├── data/
│   ├── news/                   # News dataset
│   └── opinion/                # Opinion dataset
├── graphs/                    
├── models/
│   ├── multiclass_bert/        # Fine-tuned BERT model for AI model attribution
│   ├── binary_classifier.pkl   # Binary classifier (Human vs AI)
│   └── tfidf_vectorizer.pkl    # TF-IDF vectorizer for text preprocessing
├── results/                    # Results file (of type .txt)
├── scripts/
│   ├── model_trainer.py        # Model training script
│   ├── news_ai_generator.py    # AI text generation for news articles
│   ├── log_model_eval.py       # Script for evaluating model performance
│   └── news_visualize.py       # Visualizing results for news datasets
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies

```

## Usage
- Install dependencies: pip3 install -r requirements.txt
- To run the flask app: python app.py and open http://127.0.0.1:5000/ in your local browser.

Note: I used ChatGPT to assist me with the writing of this project.
 
