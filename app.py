from flask import Flask, request, render_template_string
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

MODEL_PATH = "fake_news_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"

# Load LSTM model and tokenizer
model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

max_len = 150

def load_html():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.route('/')
def home():
    return render_template_string(load_html())

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    seq = tokenizer.texts_to_sequences([news])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(padded)[0][0]
    result = "✅ This news is Real." if prediction < 0.5 else "🚫 This news is Fake."
    return render_template_string(load_html(), prediction_text=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
