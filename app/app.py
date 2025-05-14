from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Load the binary classifier (human vs AI) and vectorizer
binary_clf = joblib.load('models/binary_classifier.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Load the multiclass classifier (model attribution)
multiclass_model = AutoModelForSequenceClassification.from_pretrained('models/multiclass_bert')
multiclass_tokenizer = AutoTokenizer.from_pretrained('models/multiclass_bert')

# Function for binary classification prediction (Human vs AI)
def predict_binary(text):
    text_vector = vectorizer.transform([text])
    prediction = binary_clf.predict(text_vector)
    return prediction[0]

# Function for multiclass classification prediction (Model Attribution)
def predict_multiclass(text):
    inputs = multiclass_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = multiclass_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# Route for landing page
@app.route('/')
def home():
    return render_template('landing_page.html')

# Route for the analysis form
@app.route('/analyze')
def analyze():
    return render_template('index.html')  # The page where the user submits text for analysis


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    
    # Predict binary classification (Human vs AI)
    binary_result = predict_binary(text)
    
    # Predict multiclass classification (AI Model Attribution)
    multiclass_result = predict_multiclass(text)
    
    # Map the multiclass result to model names
    models = ["GPT-4o", "Claude", "Mistral"]
    if binary_result == 0:  # If "Human" is detected, avoid model attribution
        multiclass_label = "N/A"
    else:
        multiclass_label = models[multiclass_result]
    
    return jsonify({
        'binary_result': 'AI' if binary_result == 1 else 'Human',
        'multiclass_result': multiclass_label
    })

if __name__ == "__main__":
    app.run(debug=True)
