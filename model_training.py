"""
Data Preprocessing & Machine Learning Training Pipeline
ITAI - AI in Finance (Option 1 Deliverable)
"""

import re
import string
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


def clean_text(text: str) -> str:
    """Standardizes financial text: removes URLs, punctuation, and lowercases."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.lower().strip()


def train_sentiment_model(csv_path: str = "sample_data.csv",
                          model_out: str = "sentiment_model.pkl",
                          vectorizer_out: str = "vectorizer.pkl"):
    """Trains a MultinomialNB sentiment classifier and saves artifacts."""
    print(f"[*] Loading training corpus from: {csv_path}")
    df = pd.read_csv(csv_path)
    df["cleaned_text"] = df["text"].apply(clean_text)

    X = df["cleaned_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = MultinomialNB(alpha=0.5)
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)

    print(f"[✓] Model Training Complete. Test Accuracy: {acc * 100:.2f}%")
    print("\n--- Classification Performance Report ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(clf, model_out)
    joblib.dump(vectorizer, vectorizer_out)
    print(f"[✓] Artifacts persisted: '{model_out}' and '{vectorizer_out}'")


def load_sentiment_pipeline(model_path: str = "sentiment_model.pkl",
                            vectorizer_path: str = "vectorizer.pkl"):
    """Loads artifacts and returns a callable inference function."""
    clf = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    def predict_sentiment(raw_text: str) -> str:
        cleaned = clean_text(raw_text)
        features = vectorizer.transform([cleaned])
        return clf.predict(features)[0]

    return predict_sentiment


if __name__ == "__main__":
    train_sentiment_model()
