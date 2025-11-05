# train_model.py
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

# ---------- Helpers ----------
def is_english(text):
    """Return True if text contains majority English letters."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return False
    english_chars = len(re.findall(r'[A-Za-z]', text))
    return english_chars / max(len(text), 1) > 0.5

def find_text_column(df):
    """Try to find the best candidate text column (long strings)."""
    # Candidate columns with average length > 20
    candidates = [c for c in df.columns if df[c].astype(str).str.len().mean() > 20]
    if candidates:
        return candidates[0]
    # fallback: the longest-mean column
    lengths = {c: df[c].astype(str).str.len().mean() for c in df.columns}
    return max(lengths, key=lengths.get)

def find_label_column(df):
    """Try to find label column with small number of unique string values."""
    for c in df.columns:
        if df[c].dtype == object and 2 <= df[c].nunique() <= 10:
            return c
    # fallback: any column named like 'label' or 'target'
    for c in df.columns:
        if c.lower() in ('label', 'labels', 'target', 'class', 'category'):
            return c
    return None

# ---------- Load news dataset ----------
news_path = 'bharatfakenewskosh.xlsx'
news_df = pd.read_excel(news_path)

# Detect columns
text_col = None
label_col = None

# Prefer direct names if exist
for prefer in ('text', 'content', 'article', 'news', 'body'):
    if prefer in news_df.columns:
        text_col = prefer
        break
for prefer in ('label', 'labels', 'target', 'class', 'category'):
    if prefer in news_df.columns:
        label_col = prefer
        break

# If not found, heuristics
if text_col is None:
    text_col = find_text_column(news_df)
if label_col is None:
    label_col = find_label_column(news_df)

if text_col is None or label_col is None:
    print("Available columns in news file:", list(news_df.columns))
    raise Exception("Could not auto-detect text or label column in the news Excel file. Update file or script.")

# Extract relevant columns and rename
news = news_df[[text_col, label_col]].copy()
news.columns = ['text', 'label']

# Filter to English-only rows
news['is_english'] = news['text'].apply(is_english)
news_en = news[news['is_english']].drop(columns=['is_english']).reset_index(drop=True)

print(f"News: original={len(news)}, kept_english={len(news_en)}")

# ---------- Load neutral facts ----------
neutral_path = 'neutral_facts.csv'
if not os.path.exists(neutral_path):
    raise FileNotFoundError(f"Neutral facts file not found: {neutral_path}")

neutral_df = pd.read_csv(neutral_path)

# Attempt to find text column in neutral file (common names)
neutral_text_col = None
for prefer in ('text', 'fact', 'content', 'sentence', 'statement'):
    if prefer in neutral_df.columns:
        neutral_text_col = prefer
        break
if neutral_text_col is None:
    neutral_text_col = find_text_column(neutral_df)

neutral = neutral_df[[neutral_text_col]].copy()
neutral.columns = ['text']
neutral['label'] = 'REAL'  # neutral facts are labelled REAL

# Filter neutral to English as well
neutral['is_english'] = neutral['text'].apply(is_english)
neutral_en = neutral[neutral['is_english']].drop(columns=['is_english']).reset_index(drop=True)

print(f"Neutral: original={len(neutral)}, kept_english={len(neutral_en)}")

# ---------- Combine datasets ----------
data = pd.concat([news_en, neutral_en], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
print(f"Combined dataset size (after shuffle): {len(data)}")
print(f"Label distribution:\n{data['label'].value_counts()}")

# ---------- Train/Test split ----------
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------- Vectorize & Train ----------
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ---------- Evaluate ----------
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained. Test Accuracy: {acc*100:.2f}%")

# ---------- Save artifacts ----------
os.makedirs('model', exist_ok=True)
pickle.dump(model, open('model/fake_news_model.pkl', 'wb'))
pickle.dump(vectorizer, open('model/vectorizer.pkl', 'wb'))
print("✅ Saved model -> model/fake_news_model.pkl")
print("✅ Saved vectorizer -> model/vectorizer.pkl")
