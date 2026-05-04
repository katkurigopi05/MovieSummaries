"""
evaluation.py — Model evaluation, metrics, and chart generation.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)

# ──── style ────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def evaluate_model(model, X, Y_true, label_names: list) -> dict:
    """
    Evaluate a multi-label model.

    Returns a dict with overall and per-class metrics.
    """
    Y_pred = model.predict(X)

    results = {
        "subset_accuracy": float(accuracy_score(Y_true, Y_pred)),
        "hamming_loss": float(hamming_loss(Y_true, Y_pred)),
        "micro_precision": float(precision_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(Y_true, Y_pred, average="weighted", zero_division=0)),
    }

    # Per-class report
    report = classification_report(
        Y_true, Y_pred, target_names=label_names, zero_division=0, output_dict=True
    )
    results["per_class"] = report

    return results


def print_results(name: str, results: dict):
    """Pretty-print evaluation results."""
    print(f"\n{'═' * 60}")
    print(f"  {name}")
    print(f"{'═' * 60}")
    print(f"  Subset Accuracy : {results['subset_accuracy']:.4f}")
    print(f"  Hamming Loss    : {results['hamming_loss']:.4f}")
    print(f"  Micro F1        : {results['micro_f1']:.4f}")
    print(f"  Macro F1        : {results['macro_f1']:.4f}")
    print(f"  Weighted F1     : {results['weighted_f1']:.4f}")
    print(f"{'─' * 60}")


def save_metrics(results: dict, path: str):
    """Save evaluation results as JSON."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Metrics saved → {path}")


# ──────────────────────────── Charts ────────────────────────────


def plot_genre_distribution(df, save_path: str):
    """Bar chart of genre frequencies (EDA)."""
    from collections import Counter

    counter = Counter()
    for gl in df["genres"]:
        counter.update(gl)

    genres, counts = zip(*counter.most_common())
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("viridis", len(genres))
    ax.barh(list(reversed(genres)), list(reversed(counts)), color=list(reversed(colors)))
    ax.set_xlabel("Number of Movies")
    ax.set_title("Genre Distribution (Top Genres)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Chart saved → {save_path}")


def plot_summary_length_distribution(df, save_path: str):
    """Histogram of plot summary lengths (EDA)."""
    lengths = df["plot"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=80, color="#5E81AC", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Number of Words")
    ax.set_ylabel("Number of Movies")
    ax.set_title("Plot Summary Length Distribution", fontsize=14, fontweight="bold")
    ax.axvline(lengths.median(), color="#BF616A", linestyle="--", label=f"Median = {int(lengths.median())}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Chart saved → {save_path}")


def plot_genres_per_movie(df, save_path: str):
    """Histogram showing how many genres each movie has."""
    n_genres = df["genres"].apply(len)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(n_genres, bins=range(1, n_genres.max() + 2), color="#A3BE8C", edgecolor="white", align="left")
    ax.set_xlabel("Number of Genres per Movie")
    ax.set_ylabel("Count")
    ax.set_title("Genres per Movie Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Chart saved → {save_path}")


def plot_confusion_matrices(model, X, Y_true, label_names, save_path: str):
    """Per-class confusion matrix heatmaps in a grid."""
    Y_pred = model.predict(X)
    cms = multilabel_confusion_matrix(Y_true, Y_pred)

    n = len(label_names)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3))
    axes = axes.flatten()

    for idx, (cm, name) in enumerate(zip(cms, label_names)):
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"],
            cbar=False,
        )
        axes[idx].set_title(name, fontsize=9, fontweight="bold")
        axes[idx].set_ylabel("True")
        axes[idx].set_xlabel("Pred")

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Per-Genre Confusion Matrices", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved → {save_path}")


def plot_model_comparison(all_results: dict, save_path: str):
    """Grouped bar chart comparing models on key metrics."""
    metrics = ["micro_f1", "macro_f1", "weighted_f1", "subset_accuracy"]
    labels = list(all_results.keys())

    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#5E81AC", "#A3BE8C", "#BF616A"]
    for i, (name, res) in enumerate(all_results.items()):
        vals = [res[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=name, color=colors[i % 3])
        # Value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Chart saved → {save_path}")


def plot_per_genre_f1(all_results: dict, label_names: list, save_path: str):
    """Grouped horizontal bar chart of per-genre F1 for each model."""
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(label_names))
    height = 0.25
    colors = ["#5E81AC", "#A3BE8C", "#BF616A"]

    for i, (model_name, res) in enumerate(all_results.items()):
        per_class = res["per_class"]
        f1_vals = [per_class.get(g, {}).get("f1-score", 0) for g in label_names]
        ax.barh(y + i * height, f1_vals, height, label=model_name, color=colors[i % 3])

    ax.set_yticks(y + height)
    ax.set_yticklabels(label_names, fontsize=9)
    ax.set_xlabel("F1-Score")
    ax.set_title("Per-Genre F1-Score by Model", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Chart saved → {save_path}")
