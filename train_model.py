# train_model.py
import os
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample

# ---------- CONFIG ----------
DATA_FILE = "india_fake_news_dataset.csv"   # generated dataset (20k entries)
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
RANDOM_STATE = 42
# ----------------------------

def clean_text(s):
    s = str(s)
    s = s.lower()
    s = re.sub(r"http\S+|www\S+|https\S+", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_dataset(path):
    df = pd.read_csv(path)
    # Expect columns: text,label (0/1 or 1/0)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must have 'text' and 'label' columns.")
    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).apply(clean_text)
    # Ensure labels are ints 0/1
    df["label"] = df["label"].astype(str).str.strip().str.lower().replace({
        "real":"1","true":"1","yes":"1",
        "fake":"0","false":"0","no":"0"
    })
    # numeric conversion
    df["label"] = pd.to_numeric(df["label"], errors='coerce')
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df

def balance_dataset(df):
    # Ensure classes are 0 and 1; if unbalanced, downsample majority to minority size
    counts = df["label"].value_counts()
    if len(counts) < 2:
        raise ValueError("Dataset must contain both classes (0 and 1).")
    min_size = counts.min()
    dfs = []
    for lbl in sorted(df["label"].unique()):
        part = df[df["label"] == lbl]
        if len(part) > min_size:
            part = resample(part, replace=False, n_samples=min_size, random_state=RANDOM_STATE)
        dfs.append(part)
    balanced = pd.concat(dfs).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return balanced

def main():
    print("Loading dataset:", DATA_FILE)
    df = load_dataset(DATA_FILE)
    print("Original class distribution:")
    print(df["label"].value_counts())

    df = balance_dataset(df)
    print("Balanced class distribution:")
    print(df["label"].value_counts())

    X = df["text"].astype(str)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=15000, stop_words="english", ngram_range=(1,2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)
    model.fit(X_train_tfidf, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"\n✅ Model and vectorizer saved to '{MODEL_DIR}/'")

if __name__ == "__main__":
    main()
