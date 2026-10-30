"""Train and compare spam classifiers, then save the best model.

Usage:  python train.py
"""
import io
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless backend, works without a display
import matplotlib.pyplot as plt
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "sms.tsv"
MODEL_PATH = BASE / "models" / "spam_model.joblib"
METRICS_PATH = BASE / "models" / "metrics.json"
CM_PATH = BASE / "models" / "confusion_matrix.png"
DATA_URL = ("https://raw.githubusercontent.com/justmarkham/"
            "pycon-2016-tutorial/master/data/sms.tsv")
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    """Load the SMS spam dataset, downloading it on first run."""
    if not DATA_PATH.exists():
        print(f"Downloading dataset from {DATA_URL} ...")
        resp = requests.get(DATA_URL, timeout=30)
        resp.raise_for_status()
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        DATA_PATH.write_bytes(resp.content)
    df = pd.read_csv(DATA_PATH, sep="\t", header=None,
                     names=["label", "message"], encoding="utf-8")
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df["target"] = (df["label"] == "spam").astype(int)
    return df


def build_models() -> dict:
    def tfidf():
        return TfidfVectorizer(lowercase=True, stop_words="english",
                               ngram_range=(1, 2), min_df=2)
    return {
        "Naive Bayes": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB())]),
        "Logistic Regression": Pipeline([
            ("tfidf", tfidf()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", tfidf()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                           n_jobs=-1)),
        ]),
    }


def main() -> None:
    df = load_data()
    print(f"Loaded {len(df)} messages ({df['target'].sum()} spam).")

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["target"], test_size=0.2,
        random_state=RANDOM_STATE, stratify=df["target"])

    results, fitted = {}, {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        }
        fitted[name] = model
        print(f"{name:20s} {results[name]}")

    best_name = max(results, key=lambda n: results[n]["f1"])
    best = fitted[best_name]
    print(f"\nBest model: {best_name}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({"best_model": best_name,
                                        "results": results}, indent=2))

    cm = confusion_matrix(y_test, best.predict(X_test))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
