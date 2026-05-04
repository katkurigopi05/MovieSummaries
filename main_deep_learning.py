#!/usr/bin/env python3
"""
main_deep_learning.py — Deep Learning pipeline for Movie Genre Classification.

Trains BiLSTM and DistilBERT models, evaluates them on the test set,
and creates a combined 5-model comparison with the existing ML models.

Steps:
  1. Load & merge data (reuses src/data_loader.py)
  2. Filter top genres
  3. Preprocess text (reuses src/preprocessing.py)
  4. Train/Validation/Test split
  5. Train BiLSTM (GloVe + Self-Attention)
  6. Train DistilBERT (fine-tuned)
  7. Evaluate both on test set
  8. Load ML results and create 5-model comparison
  9. Save models, metrics, and charts
 10. Demo inference
"""

import os
import sys
import time
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# ── Project modules ──
from src.data_loader import load_metadata, load_plots, merge_data, filter_top_genres
from src.preprocessing import preprocess_dataframe, preprocess_text
from src.lstm_model import (
    build_vocab,
    train_lstm,
    predict_lstm,
)
from src.bert_model import train_bert, predict_bert
from src.dl_evaluation import (
    evaluate_dl_predictions,
    print_dl_results,
    save_dl_metrics,
    plot_training_curves,
    plot_all_models_comparison,
    plot_all_per_genre_f1,
    plot_dl_confusion_matrices,
)
from src.dl_inference import predict_genres_lstm, predict_genres_bert

# ── Paths ──
META_PATH = "movie.metadata.tsv"
PLOT_PATH = "plot_summaries.txt"
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
FIG_DIR = "outputs/figures"
METRIC_DIR = "outputs/metrics"

TOP_N_GENRES = 20
RANDOM_STATE = 42


def banner(msg: str):
    print(f"\n{'━' * 64}")
    print(f"  ▶  {msg}")
    print(f"{'━' * 64}")


def main():
    t0 = time.time()

    # ── 1. Load data ────────────────────────────────────────────
    banner("Step 1 · Loading data")
    meta_df = load_metadata(META_PATH)
    plot_df = load_plots(PLOT_PATH)
    merged = merge_data(meta_df, plot_df)
    print(f"  Metadata rows : {len(meta_df):,}")
    print(f"  Plot rows     : {len(plot_df):,}")
    print(f"  Merged rows   : {len(merged):,}")

    # ── 2. Filter top genres ────────────────────────────────────
    banner("Step 2 · Filtering top genres")
    df, genre_names = filter_top_genres(merged, top_n=TOP_N_GENRES)
    print(f"  Keeping top {TOP_N_GENRES} genres → {len(df):,} movies remain")
    print(f"  Genres: {genre_names}")

    # ── 3. Preprocess text ──────────────────────────────────────
    banner("Step 3 · Preprocessing text")
    df = preprocess_dataframe(df, text_col="plot", new_col="clean_plot")

    # ── 4. Train / Validation / Test split ──────────────────────
    banner("Step 4 · Splitting data (70/15/15)")
    mlb = MultiLabelBinarizer(classes=genre_names)
    Y = mlb.fit_transform(df["genres"])
    X_texts = df["clean_plot"].values

    # Use raw plots for BERT (BERT has its own tokenizer)
    X_raw_texts = df["plot"].values

    X_train_text, X_temp_text, Y_train, Y_temp = train_test_split(
        X_texts, Y, test_size=0.30, random_state=RANDOM_STATE
    )
    X_val_text, X_test_text, Y_val, Y_test = train_test_split(
        X_temp_text, Y_temp, test_size=0.50, random_state=RANDOM_STATE
    )

    # Same split for raw texts (BERT)
    X_train_raw, X_temp_raw, _, _ = train_test_split(
        X_raw_texts, Y, test_size=0.30, random_state=RANDOM_STATE
    )
    X_val_raw, X_test_raw, _, _ = train_test_split(
        X_temp_raw, Y_temp, test_size=0.50, random_state=RANDOM_STATE
    )

    print(f"  Train : {len(X_train_text):,}")
    print(f"  Val   : {len(X_val_text):,}")
    print(f"  Test  : {len(X_test_text):,}")
    print(f"  Labels: {Y_train.shape[1]} genres")

    # Ensure model/output dirs exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(METRIC_DIR, exist_ok=True)

    # Save label binarizer (overwrite if exists)
    joblib.dump(mlb, os.path.join(MODEL_DIR, "label_binarizer.joblib"))

    # ================================================================
    # 5. TRAIN BiLSTM
    # ================================================================
    banner("Step 5 · Training BiLSTM (GloVe + Self-Attention)")
    t1 = time.time()

    lstm_model, vocab, lstm_history = train_lstm(
        train_texts=X_train_text,
        train_labels=Y_train,
        val_texts=X_val_text,
        val_labels=Y_val,
        num_labels=len(genre_names),
        max_len=500,
        embed_dim=100,
        hidden_dim=128,
        batch_size=64,
        epochs=5,
        lr=1e-3,
        patience=2,
        model_dir=MODEL_DIR,
        use_glove=True,
    )

    lstm_train_time = time.time() - t1
    print(f"  BiLSTM trained in {lstm_train_time:.0f}s")

    # Plot LSTM training curves
    plot_training_curves(
        lstm_history, "BiLSTM",
        os.path.join(FIG_DIR, "lstm_training_curves.png"),
    )

    # ================================================================
    # 6. TRAIN DistilBERT
    # ================================================================
    banner("Step 6 · Training DistilBERT (Fine-tuned)")
    t1 = time.time()

    bert_model, bert_tokenizer, bert_history = train_bert(
        train_texts=X_train_raw,
        train_labels=Y_train,
        val_texts=X_val_raw,
        val_labels=Y_val,
        num_labels=len(genre_names),
        max_len=256,
        batch_size=16,
        epochs=3,
        lr=2e-5,
        grad_accum_steps=4,
        freeze_layers=4,
        patience=2,
        model_dir=MODEL_DIR,
    )

    bert_train_time = time.time() - t1
    print(f"  DistilBERT trained in {bert_train_time:.0f}s")

    # Plot BERT training curves
    plot_training_curves(
        bert_history, "DistilBERT",
        os.path.join(FIG_DIR, "bert_training_curves.png"),
    )

    # ================================================================
    # 7. EVALUATE on test set
    # ================================================================
    banner("Step 7 · Evaluating on TEST set")

    # LSTM predictions
    print("\n  ── BiLSTM ──")
    lstm_preds = predict_lstm(lstm_model, X_test_text, vocab, threshold=0.3)
    lstm_results = evaluate_dl_predictions(Y_test, lstm_preds, genre_names)
    print_dl_results("BiLSTM (Test)", lstm_results)

    # BERT predictions
    print("\n  ── DistilBERT ──")
    bert_preds = predict_bert(bert_model, X_test_raw, bert_tokenizer, threshold=0.3)
    bert_results = evaluate_dl_predictions(Y_test, bert_preds, genre_names)
    print_dl_results("DistilBERT (Test)", bert_results)

    # ================================================================
    # 8. COMBINED 5-MODEL COMPARISON
    # ================================================================
    banner("Step 8 · Creating 5-model comparison")

    # Try to load existing ML results
    all_results = {}
    ml_metrics_files = {
        "Naive Bayes": "best_model_test_metrics.json",  # We'll load individual ones
        "Logistic Regression": "logistic_regression_val_metrics.json",
        "Linear SVM": "linear_svm_val_metrics.json",
    }

    # Try to load ML test results from the summary
    summary_path = os.path.join(METRIC_DIR, "model_comparison_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            ml_summary = json.load(f)

        # Load individual validation metrics for per-class data
        for ml_name in ["Naive Bayes", "Logistic Regression", "Linear SVM"]:
            fname = ml_name.lower().replace(" ", "_") + "_val_metrics.json"
            fpath = os.path.join(METRIC_DIR, fname)
            if os.path.exists(fpath):
                with open(fpath, "r") as f:
                    all_results[ml_name] = json.load(f)
                print(f"  Loaded ML metrics: {ml_name}")

    # Add DL results
    all_results["BiLSTM"] = lstm_results
    all_results["DistilBERT"] = bert_results

    if len(all_results) >= 3:
        print(f"  Total models for comparison: {len(all_results)}")

        # Combined comparison chart
        plot_all_models_comparison(
            all_results,
            os.path.join(FIG_DIR, "all_models_comparison.png"),
        )

        # Per-genre F1 comparison
        plot_all_per_genre_f1(
            all_results, genre_names,
            os.path.join(FIG_DIR, "all_per_genre_f1.png"),
        )
    else:
        print("  ⚠ ML results not found — run main.py first for full comparison")
        # Still create DL-only comparison
        dl_results = {"BiLSTM": lstm_results, "DistilBERT": bert_results}
        plot_all_models_comparison(
            dl_results,
            os.path.join(FIG_DIR, "dl_models_comparison.png"),
        )

    # Confusion matrices for DL models
    plot_dl_confusion_matrices(
        Y_test, lstm_preds, genre_names,
        os.path.join(FIG_DIR, "confusion_matrices_bilstm.png"),
    )
    plot_dl_confusion_matrices(
        Y_test, bert_preds, genre_names,
        os.path.join(FIG_DIR, "confusion_matrices_distilbert.png"),
    )

    # ================================================================
    # 9. SAVE METRICS
    # ================================================================
    banner("Step 9 · Saving metrics")

    save_dl_metrics(lstm_results, os.path.join(METRIC_DIR, "lstm_test_metrics.json"))
    save_dl_metrics(bert_results, os.path.join(METRIC_DIR, "bert_test_metrics.json"))

    # Save combined summary
    dl_summary = {
        "lstm": {
            "macro_f1": lstm_results["macro_f1"],
            "micro_f1": lstm_results["micro_f1"],
            "weighted_f1": lstm_results["weighted_f1"],
            "subset_accuracy": lstm_results["subset_accuracy"],
            "training_time_seconds": lstm_train_time,
        },
        "bert": {
            "macro_f1": bert_results["macro_f1"],
            "micro_f1": bert_results["micro_f1"],
            "weighted_f1": bert_results["weighted_f1"],
            "subset_accuracy": bert_results["subset_accuracy"],
            "training_time_seconds": bert_train_time,
        },
    }

    # Determine best DL model
    if bert_results["macro_f1"] >= lstm_results["macro_f1"]:
        dl_summary["best_dl_model"] = "DistilBERT"
    else:
        dl_summary["best_dl_model"] = "BiLSTM"

    save_dl_metrics(dl_summary, os.path.join(METRIC_DIR, "dl_comparison_summary.json"))

    # ================================================================
    # 10. DEMO INFERENCE
    # ================================================================
    banner("Step 10 · Demo inference")

    samples = [
        "A group of astronauts travel to a distant planet where they discover "
        "an alien civilization that threatens to destroy Earth. They must use "
        "advanced weapons and clever strategy to survive.",

        "A young woman moves to New York City and falls in love with her "
        "neighbor. Through a series of comedic misunderstandings, they "
        "eventually realize they are meant to be together.",

        "A detective investigates a series of grisly murders in a small "
        "town. As the body count rises, he realizes the killer is someone "
        "he knows and must confront his own dark past.",
    ]

    for i, text in enumerate(samples, 1):
        print(f"\n  Sample {i}: \"{text[:80]}…\"")

        lstm_genres = predict_genres_lstm(text, lstm_model, vocab, mlb, threshold=0.3)
        print(f"  → BiLSTM:      {lstm_genres}")

        bert_genres = predict_genres_bert(text, bert_model, bert_tokenizer, mlb, threshold=0.3)
        print(f"  → DistilBERT:  {bert_genres}")

    # ── Done ────────────────────────────────────────────────────
    elapsed = time.time() - t0
    banner(f"Deep Learning pipeline complete in {elapsed:.0f}s")
    print(f"  Models  → {MODEL_DIR}/")
    print(f"  Charts  → {FIG_DIR}/")
    print(f"  Metrics → {METRIC_DIR}/")
    print()

    # Print final summary table
    print("  ┌─────────────────┬───────────┬───────────┬───────────┐")
    print("  │ Model           │ Macro F1  │ Micro F1  │ Train (s) │")
    print("  ├─────────────────┼───────────┼───────────┼───────────┤")
    print(f"  │ BiLSTM          │  {lstm_results['macro_f1']:.4f}   │  {lstm_results['micro_f1']:.4f}   │  {lstm_train_time:>7.0f}  │")
    print(f"  │ DistilBERT      │  {bert_results['macro_f1']:.4f}   │  {bert_results['micro_f1']:.4f}   │  {bert_train_time:>7.0f}  │")
    print("  └─────────────────┴───────────┴───────────┴───────────┘")
    print()


if __name__ == "__main__":
    main()
