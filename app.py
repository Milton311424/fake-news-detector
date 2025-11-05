from flask import Flask, request, render_template_string
import pickle
import os

app = Flask(__name__)

# ✅ Model and vectorizer in root folder
MODEL_PATH = "fake_news_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# Load model and vectorizer
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# ✅ Load HTML directly from the root index.html file
def load_html():
    with open("index.html", "r", encoding="utf-8") as file:
        return file.read()

@app.route('/')
def home():
    html = load_html()
    return render_template_string(html)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news']
        if not news_text.strip():
            return render_template_string(load_html(), prediction_text="⚠️ Please enter some text.")
        
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)[0]

        result = "✅ This news is Real." if prediction == 0 else "🚫 This news is Fake."
        return render_template_string(load_html(), prediction_text=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
