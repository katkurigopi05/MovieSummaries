#!/usr/bin/env python3
"""
main.py — End-to-end pipeline for Movie Genre Classification.

Steps:
  1. Load & merge data
  2. Filter top genres
  3. Preprocess text
  4. Exploratory Data Analysis (EDA charts)
  5. Train/Validation/Test split
  6. TF-IDF feature extraction
  7. Train models (Naive Bayes, Logistic Regression, Linear SVM)
  8. Evaluate on validation set → pick best model
  9. Final evaluation on held-out test set
 10. Save best model, vectorizer, binarizer, metrics, and charts
 11. Demo inference on sample texts
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
from src.preprocessing import preprocess_dataframe
from src.feature_engineering import build_tfidf, transform_tfidf, save_vectorizer
from src.model_training import (
    train_naive_bayes,
    train_logistic_regression,
    train_linear_svm,
    save_model,
)
from src.evaluation import (
    evaluate_model,
    print_results,
    save_metrics,
    plot_genre_distribution,
    plot_summary_length_distribution,
    plot_genres_per_movie,
    plot_confusion_matrices,
    plot_model_comparison,
    plot_per_genre_f1,
)
from src.inference import predict_genres

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

    # ── 1. Load data ────────────────────────────────────────────────
    banner("Step 1 · Loading data")
    meta_df = load_metadata(META_PATH)
    plot_df = load_plots(PLOT_PATH)
    merged = merge_data(meta_df, plot_df)
    print(f"  Metadata rows : {len(meta_df):,}")
    print(f"  Plot rows     : {len(plot_df):,}")
    print(f"  Merged rows   : {len(merged):,}")

    # ── 2. Filter top genres ────────────────────────────────────────
    banner("Step 2 · Filtering top genres")
    df, genre_names = filter_top_genres(merged, top_n=TOP_N_GENRES)
    print(f"  Keeping top {TOP_N_GENRES} genres → {len(df):,} movies remain")
    print(f"  Genres: {genre_names}")

    # ── 3. Preprocess text ──────────────────────────────────────────
    banner("Step 3 · Preprocessing text")
    df = preprocess_dataframe(df, text_col="plot", new_col="clean_plot")

    # Save processed data
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    save_path = os.path.join(PROCESSED_DIR, "processed_data.csv")
    df_save = df[["wiki_id", "name", "clean_plot", "genres"]].copy()
    df_save["genres"] = df_save["genres"].apply(json.dumps)
    df_save.to_csv(save_path, index=False)
    print(f"  Processed data saved → {save_path}")

    # ── 4. EDA charts ───────────────────────────────────────────────
    banner("Step 4 · Generating EDA charts")
    os.makedirs(FIG_DIR, exist_ok=True)
    plot_genre_distribution(df, os.path.join(FIG_DIR, "genre_distribution.png"))
    plot_summary_length_distribution(df, os.path.join(FIG_DIR, "summary_length_distribution.png"))
    plot_genres_per_movie(df, os.path.join(FIG_DIR, "genres_per_movie.png"))

    # ── 5. Train / Validation / Test split ──────────────────────────
    banner("Step 5 · Splitting data (70/15/15)")
    mlb = MultiLabelBinarizer(classes=genre_names)
    Y = mlb.fit_transform(df["genres"])
    X_texts = df["clean_plot"].values

    # First split: 70% train, 30% temp
    X_train_text, X_temp_text, Y_train, Y_temp = train_test_split(
        X_texts, Y, test_size=0.30, random_state=RANDOM_STATE
    )
    # Second split: 50/50 of temp → 15% val, 15% test
    X_val_text, X_test_text, Y_val, Y_test = train_test_split(
        X_temp_text, Y_temp, test_size=0.50, random_state=RANDOM_STATE
    )
    print(f"  Train : {len(X_train_text):,}")
    print(f"  Val   : {len(X_val_text):,}")
    print(f"  Test  : {len(X_test_text):,}")
    print(f"  Labels: {Y_train.shape[1]} genres")

    # Save label binarizer
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(mlb, os.path.join(MODEL_DIR, "label_binarizer.joblib"))

    # ── 6. TF-IDF feature extraction ───────────────────────────────
    banner("Step 6 · Building TF-IDF features")
    vectorizer, X_train = build_tfidf(X_train_text)
    X_val = transform_tfidf(vectorizer, X_val_text)
    X_test = transform_tfidf(vectorizer, X_test_text)
    save_vectorizer(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))

    # ── 7. Train models ────────────────────────────────────────────
    models = {}

    banner("Step 7a · Training Naive Bayes")
    t1 = time.time()
    models["Naive Bayes"] = train_naive_bayes(X_train, Y_train)
    print(f"  Trained in {time.time() - t1:.1f}s")

    banner("Step 7b · Training Logistic Regression")
    t1 = time.time()
    models["Logistic Regression"] = train_logistic_regression(X_train, Y_train)
    print(f"  Trained in {time.time() - t1:.1f}s")

    banner("Step 7c · Training Linear SVM")
    t1 = time.time()
    models["Linear SVM"] = train_linear_svm(X_train, Y_train)
    print(f"  Trained in {time.time() - t1:.1f}s")

    # ── 8. Evaluate on validation set ──────────────────────────────
    banner("Step 8 · Evaluating on validation set")
    all_val_results = {}
    for name, model in models.items():
        results = evaluate_model(model, X_val, Y_val, genre_names)
        all_val_results[name] = results
        print_results(f"{name} (Validation)", results)

    # Pick best by macro F1
    best_name = max(all_val_results, key=lambda k: all_val_results[k]["macro_f1"])
    print(f"\n  🏆 Best model (macro F1): {best_name}")

    # ── 9. Final evaluation on test set ────────────────────────────
    banner("Step 9 · Final evaluation on TEST set")
    best_model = models[best_name]
    test_results = evaluate_model(best_model, X_test, Y_test, genre_names)
    print_results(f"{best_name} (Test)", test_results)

    # ── 10. Save everything ────────────────────────────────────────
    banner("Step 10 · Saving models, metrics, and charts")

    # Save all models
    for name, model in models.items():
        fname = name.lower().replace(" ", "_") + ".joblib"
        save_model(model, os.path.join(MODEL_DIR, fname))

    # Save best model with canonical name
    save_model(best_model, os.path.join(MODEL_DIR, "best_model.joblib"))

    # Save metrics
    os.makedirs(METRIC_DIR, exist_ok=True)
    for name, res in all_val_results.items():
        fname = name.lower().replace(" ", "_") + "_val_metrics.json"
        save_metrics(res, os.path.join(METRIC_DIR, fname))
    save_metrics(test_results, os.path.join(METRIC_DIR, "best_model_test_metrics.json"))

    # Save summary comparison
    summary = {
        "best_model": best_name,
        "validation_results": {
            k: {m: v[m] for m in ["micro_f1", "macro_f1", "weighted_f1", "subset_accuracy"]}
            for k, v in all_val_results.items()
        },
        "test_results": {
            m: test_results[m]
            for m in ["micro_f1", "macro_f1", "weighted_f1", "subset_accuracy"]
        },
    }
    save_metrics(summary, os.path.join(METRIC_DIR, "model_comparison_summary.json"))

    # Charts
    plot_model_comparison(all_val_results, os.path.join(FIG_DIR, "model_comparison.png"))
    plot_per_genre_f1(all_val_results, genre_names, os.path.join(FIG_DIR, "per_genre_f1.png"))

    # Confusion matrices for best model
    plot_confusion_matrices(
        best_model, X_test, Y_test, genre_names,
        os.path.join(FIG_DIR, f"confusion_matrices_{best_name.lower().replace(' ', '_')}.png"),
    )

    # ── 11. Demo inference ─────────────────────────────────────────
    banner("Step 11 · Demo inference")
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
        genres = predict_genres(text, best_model, vectorizer, mlb)
        print(f"\n  Sample {i}: \"{text[:80]}…\"")
        print(f"  → Predicted genres: {genres}")

    # ── Done ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    banner(f"Pipeline complete in {elapsed:.0f}s")
    print(f"  Models  → {MODEL_DIR}/")
    print(f"  Charts  → {FIG_DIR}/")
    print(f"  Metrics → {METRIC_DIR}/")
    print()


if __name__ == "__main__":
    main()
