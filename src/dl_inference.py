"""
dl_inference.py — Unified deep learning inference for movie genre prediction.

Supports both LSTM and DistilBERT models with auto-detection.
"""

import json
import os

import numpy as np
import torch
import joblib

from src.preprocessing import preprocess_text


def load_lstm_artifacts(model_dir: str = "models"):
    """
    Load LSTM model, vocabulary, and label binarizer from disk.

    Returns:
        model, vocab, mlb
    """
    from src.lstm_model import BiLSTMClassifier, load_vocab

    # Load label binarizer
    mlb = joblib.load(os.path.join(model_dir, "label_binarizer.joblib"))
    num_labels = len(mlb.classes_)

    # Load vocabulary
    vocab = load_vocab(os.path.join(model_dir, "lstm_vocab.json"))

    # Rebuild model architecture and load weights
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=100,
        hidden_dim=128,
        num_labels=num_labels,
    )
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "lstm_model.pt"), weights_only=True)
    )
    model.eval()

    return model, vocab, mlb


def load_bert_artifacts(model_dir: str = "models"):
    """
    Load BERT model, tokenizer, and label binarizer from disk.

    Returns:
        model, tokenizer, mlb
    """
    from src.bert_model import GenreBERT
    from transformers import DistilBertTokenizer

    # Load label binarizer
    mlb = joblib.load(os.path.join(model_dir, "label_binarizer.joblib"))
    num_labels = len(mlb.classes_)

    # Load tokenizer
    bert_dir = os.path.join(model_dir, "bert_model")
    tokenizer = DistilBertTokenizer.from_pretrained(bert_dir)

    # Rebuild model and load weights
    model = GenreBERT(num_labels=num_labels, freeze_layers=0)
    model.load_state_dict(
        torch.load(os.path.join(bert_dir, "model.pt"), weights_only=True)
    )
    model.eval()

    return model, tokenizer, mlb


def predict_genres_lstm(
    text: str,
    model,
    vocab: dict,
    mlb,
    threshold: float = 0.3,
) -> list:
    """
    Predict genres for a raw plot summary using the LSTM model.

    Args:
        text:      Raw plot summary
        model:     Trained BiLSTMClassifier
        vocab:     Word→index vocabulary
        mlb:       Fitted MultiLabelBinarizer
        threshold: Decision threshold

    Returns:
        List of predicted genre name strings.
    """
    from src.lstm_model import predict_lstm

    # Preprocess text (same pipeline as ML models)
    clean = preprocess_text(text)

    # Predict
    preds = predict_lstm(model, [clean], vocab, threshold=threshold)
    genre_names = mlb.inverse_transform(preds)[0]

    return list(genre_names)


def predict_genres_bert(
    text: str,
    model,
    tokenizer,
    mlb,
    threshold: float = 0.3,
) -> list:
    """
    Predict genres for a raw plot summary using the BERT model.

    Note: BERT uses its own tokenizer, so we only do minimal cleaning
    (the DistilBERT tokenizer handles subword tokenization).

    Args:
        text:      Raw plot summary
        model:     Trained GenreBERT
        tokenizer: DistilBertTokenizer
        mlb:       Fitted MultiLabelBinarizer
        threshold: Decision threshold

    Returns:
        List of predicted genre name strings.
    """
    from src.bert_model import predict_bert

    # For BERT, we use the raw text (BERT tokenizer handles everything)
    # But we do basic cleaning (remove HTML, etc.)
    import re
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\[\[.*?\]\]", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    preds = predict_bert(model, [text], tokenizer, threshold=threshold)
    genre_names = mlb.inverse_transform(preds)[0]

    return list(genre_names)
