import os
import pickle
from flask import Flask, render_template, request
import numpy as np

MODEL_PATH = "fake_news_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

app = Flask(__name__)

# Load model & vectorizer (fail early if missing)
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Model or vectorizer not found. Run train_model.py first and commit both files.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "").strip()
    if text == "":
        return render_template("index.html", prediction_text="⚠️ Please enter some text to analyze.")
    # transform and predict
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    # By our training: label 1 = real, 0 = fake
    if int(pred) == 1:
        result = "✅ This news is Real."
    else:
        result = "🚫 This news is Fake."
    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))  # 5050 for local, auto for Railway
    app.run(host="0.0.0.0", port=port)
