import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Load Excel dataset
data = pd.read_excel("bharatfakenewskosh.xlsx")

# Detect columns
text_col, label_col = None, None
for col in data.columns:
    if "text" in col.lower() or "content" in col.lower() or "news" in col.lower() or "statement" in col.lower():
        text_col = col
    if "label" in col.lower() or "category" in col.lower() or "class" in col.lower():
        label_col = col

if not text_col or not label_col:
    raise Exception("❌ Could not find text/label columns!")

data = data[[text_col, label_col]].dropna()

# Normalize labels
data[label_col] = data[label_col].astype(str).str.lower().replace({
    "fake": 1,
    "false": 1,
    "real": 0,
    "true": 0
})

# Load neutral facts (real examples)
neutral = pd.read_csv("neutral_facts.csv", header=None, names=["text"])
neutral["label"] = 0
print(f"Loaded {len(neutral)} neutral facts ✅")

# Combine
merged = pd.concat([
    data.rename(columns={text_col: "text", label_col: "label"}),
    neutral
], axis=0).dropna()

# ✅ Balance dataset
real_df = merged[merged["label"] == 0]
fake_df = merged[merged["label"] == 1]

min_len = min(len(real_df), len(fake_df))
balanced_df = pd.concat([
    real_df.sample(n=min_len, random_state=42),
    fake_df.sample(n=min_len, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✅ Balanced dataset: {len(balanced_df)} total samples ({min_len} real + {min_len} fake)")

# TF-IDF
X = balanced_df["text"]
y = balanced_df["label"]

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=400)
model.fit(X_train, y_train)

# Evaluate
acc = model.score(X_test, y_test)
print(f"✅ Model Accuracy: {acc:.4f}")

# Save
os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("✅ Model retrained and saved in 'model/' successfully!")
