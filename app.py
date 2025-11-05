from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

app = Flask(__name__)

# Load the trained model and vectorizer
MODEL_PATH = os.path.join("model", "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news']
        if not news_text.strip():
            return render_template('index.html', prediction_text="⚠️ Please enter some text to analyze.")

        # Transform the input text
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)[0]

        # Output label
        if prediction == 0:
            result = "✅ This news is Real."
        else:
            result = "🚫 This news is Fake."

        return render_template('index.html', prediction_text=result)


if __name__ == "__main__":
    # Render expects the app to run on port 8080
    app.run(host='0.0.0.0', port=8080)
