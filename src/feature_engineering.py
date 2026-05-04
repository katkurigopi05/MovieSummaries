"""
feature_engineering.py — TF-IDF feature extraction.
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(
    train_texts,
    max_features: int = 50_000,
    ngram_range: tuple = (1, 2),
    min_df: int = 3,
    max_df: float = 0.95,
):
    """
    Fit a TF-IDF vectorizer on training texts.

    Returns:
        vectorizer: fitted TfidfVectorizer
        X_train:    sparse matrix of TF-IDF features
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,     # apply log-normalization
        strip_accents="unicode",
    )
    X_train = vectorizer.fit_transform(train_texts)
    print(f"  TF-IDF vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"  TF-IDF matrix shape:    {X_train.shape}")
    return vectorizer, X_train


def transform_tfidf(vectorizer, texts):
    """Transform texts using an already-fitted vectorizer."""
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer, path: str):
    """Save the fitted vectorizer to disk."""
    joblib.dump(vectorizer, path)
    print(f"  Vectorizer saved → {path}")


def load_vectorizer(path: str):
    """Load a fitted vectorizer from disk."""
    return joblib.load(path)
