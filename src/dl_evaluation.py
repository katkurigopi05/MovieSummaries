"""
dl_evaluation.py — Evaluation utilities for deep learning models.

Provides:
  - DL model evaluation wrappers (convert PyTorch outputs → sklearn metrics)
  - Training curve plots (loss + F1 per epoch)
  - Combined 5-model comparison chart (3 ML + 2 DL)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
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

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ──────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────


def evaluate_dl_predictions(Y_true: np.ndarray, Y_pred: np.ndarray, label_names: list) -> dict:
    """
    Evaluate multi-label predictions (numpy arrays).
    Same output format as src/evaluation.py → evaluate_model().
    """
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

    report = classification_report(
        Y_true, Y_pred, target_names=label_names, zero_division=0, output_dict=True
    )
    results["per_class"] = report

    return results


def print_dl_results(name: str, results: dict):
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


def save_dl_metrics(results: dict, path: str):
    """Save evaluation results as JSON."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Metrics saved → {path}")


# ──────────────────────────────────────────────────────────────────
# Training Curves
# ──────────────────────────────────────────────────────────────────


def plot_training_curves(history: dict, model_name: str, save_path: str):
    """
    Plot training/validation loss and macro F1 curves.

    Args:
        history: Dict with keys 'train_loss', 'val_loss', 'val_macro_f1'
        model_name: Name for the chart title
        save_path: Where to save the PNG
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(epochs, history["train_loss"], "o-", color="#5E81AC", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "s-", color="#BF616A", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title(f"{model_name} — Loss Curves", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # F1 curve
    ax2.plot(epochs, history["val_macro_f1"], "D-", color="#A3BE8C", linewidth=2, markersize=8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1-Score")
    ax2.set_title(f"{model_name} — Validation Macro F1", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Annotate best F1
    best_epoch = np.argmax(history["val_macro_f1"]) + 1
    best_f1 = max(history["val_macro_f1"])
    ax2.annotate(
        f"Best: {best_f1:.4f}\n(Epoch {best_epoch})",
        xy=(best_epoch, best_f1),
        xytext=(best_epoch + 0.3, best_f1 - 0.02),
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#2E3440"),
        color="#2E3440",
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved → {save_path}")


# ──────────────────────────────────────────────────────────────────
# Combined Model Comparison (5 models)
# ──────────────────────────────────────────────────────────────────


def plot_all_models_comparison(all_results: dict, save_path: str):
    """
    Grouped bar chart comparing ALL models (ML + DL) on key metrics.

    Args:
        all_results: Dict mapping model_name → results dict
        save_path: Where to save the PNG
    """
    metrics = ["micro_f1", "macro_f1", "weighted_f1", "subset_accuracy"]
    model_names = list(all_results.keys())
    n_models = len(model_names)

    x = np.arange(len(metrics))
    width = 0.8 / n_models

    # Color palette: distinct colors for each model
    colors = ["#5E81AC", "#A3BE8C", "#BF616A", "#D08770", "#B48EAD"]

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, name in enumerate(model_names):
        res = all_results[name]
        vals = [res[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=name, color=colors[i % len(colors)],
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold",
            )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("All Models Comparison (ML + Deep Learning)", fontsize=15, fontweight="bold")
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved → {save_path}")


def plot_all_per_genre_f1(all_results: dict, label_names: list, save_path: str):
    """
    Horizontal bar chart of per-genre F1 for all models.
    """
    n_models = len(all_results)
    colors = ["#5E81AC", "#A3BE8C", "#BF616A", "#D08770", "#B48EAD"]

    fig, ax = plt.subplots(figsize=(14, 10))
    y = np.arange(len(label_names))
    height = 0.8 / n_models

    for i, (model_name, res) in enumerate(all_results.items()):
        per_class = res.get("per_class", {})
        f1_vals = [per_class.get(g, {}).get("f1-score", 0) for g in label_names]
        ax.barh(y + i * height, f1_vals, height, label=model_name,
                color=colors[i % len(colors)], edgecolor="white", linewidth=0.3)

    ax.set_yticks(y + height * (n_models - 1) / 2)
    ax.set_yticklabels(label_names, fontsize=9)
    ax.set_xlabel("F1-Score", fontsize=12)
    ax.set_title("Per-Genre F1-Score — All Models", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved → {save_path}")


def plot_dl_confusion_matrices(Y_true, Y_pred, label_names, save_path: str):
    """Per-class confusion matrix heatmaps for DL model predictions."""
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

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Per-Genre Confusion Matrices (Deep Learning)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved → {save_path}")
