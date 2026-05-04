"""
inference.py — Predict movie genres from a raw plot summary.
"""

import joblib
import numpy as np

from src.preprocessing import preprocess_text


def predict_genres(
    text: str,
    model,
    vectorizer,
    mlb,
    threshold: float = 0.3,
) -> list:
    """
    Predict genres for a raw plot summary string.

    Args:
        text:       Raw plot summary
        model:      Trained multi-label classifier
        vectorizer: Fitted TF-IDF vectorizer
        mlb:        Fitted MultiLabelBinarizer
        threshold:  Decision threshold for probabilities

    Returns:
        List of predicted genre name strings.
    """
    # Preprocess
    clean = preprocess_text(text)

    # Vectorize
    X = vectorizer.transform([clean])

    # Predict
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        preds = (probs >= threshold).astype(int)
        # If nothing passes threshold, take top-1
        if preds.sum() == 0:
            preds[np.argmax(probs)] = 1
        genre_names = mlb.inverse_transform(preds.reshape(1, -1))[0]
    else:
        # Fallback to hard predictions
        preds = model.predict(X)
        genre_names = mlb.inverse_transform(preds)[0]

    return list(genre_names)


def load_inference_artifacts(model_dir: str = "models"):
    """Load the best model, vectorizer, and label binarizer from disk."""
    model = joblib.load(f"{model_dir}/best_model.joblib")
    vectorizer = joblib.load(f"{model_dir}/tfidf_vectorizer.joblib")
    mlb = joblib.load(f"{model_dir}/label_binarizer.joblib")
    return model, vectorizer, mlb
