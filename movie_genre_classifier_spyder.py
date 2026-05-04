# -*- coding: utf-8 -*-
"""
Movie Genre Classification — Single-File Spyder Version
========================================================
Automatic Movie Genre Classification Using Plot Summaries
Dataset: CMU Movie Summary Corpus

Run cell-by-cell in Spyder (Ctrl+Enter) or run the entire file.
Each section is marked with '# %%' for Spyder's cell navigation.
"""

# %% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 1 · Imports & Setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import os
import re
import ssl
import json
import time
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)

# ── Fix macOS SSL issue for NLTK downloads ──
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ── Matplotlib & Seaborn style ──
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120

# ── Configuration ──
META_PATH      = "/Users/gopikrishnareddykatkuri/Downloads/MovieSummaries/movie.metadata.tsv"
PLOT_PATH      = "/Users/gopikrishnareddykatkuri/Downloads/MovieSummaries/plot_summaries.txt"
PROCESSED_DIR  = "/Users/gopikrishnareddykatkuri/Downloads/MovieSummaries/data/processed"
MODEL_DIR      = "/Users/gopikrishnareddykatkuri/Downloads/MovieSummaries/models"
FIG_DIR        = "/Users/gopikrishnareddykatkuri/Downloads/MovieSummaries/outputs/figures"
METRIC_DIR     = "/Users/gopikrishnareddykatkuri/Downloads/MovieSummaries/outputs/metrics"
TOP_N_GENRES   = 20
RANDOM_STATE   = 42

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)

STOP_WORDS  = set(stopwords.words("english"))
LEMMATIZER  = WordNetLemmatizer()

print("✓ All imports and setup complete.")


# %% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 2 · Load Data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n▶ Step 1 · Loading data…")

META_COLUMNS = [
    "wiki_id", "freebase_id", "name", "release_date",
    "revenue", "runtime", "languages", "countries", "genres_raw",
]

meta_df = pd.read_csv(
    META_PATH, sep="\t", header=None, names=META_COLUMNS,
    dtype={"wiki_id": str}, quoting=3,
)

plot_rows = []
with open(PLOT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.split("\t", maxsplit=1)
        if len(parts) == 2:
            plot_rows.append({"wiki_id": parts[0].strip(), "plot": parts[1].strip()})
plot_df = pd.DataFrame(plot_rows)

# Merge
df = pd.merge(meta_df, plot_df, on="wiki_id", how="inner")

# Parse genre JSON
def parse_genres(raw):
    try:
        return list(json.loads(raw).values())
    except (json.JSONDecodeError, TypeError):
        return []

df["genres"] = df["genres_raw"].apply(parse_genres)

# Drop empty
df = df[df["genres"].apply(len) > 0].copy()
df = df[df["plot"].str.strip().astype(bool)].copy()
df.reset_index(drop=True, inplace=True)

print(f"  Metadata rows : {len(meta_df):,}")
print(f"  Plot rows     : {len(plot_df):,}")
print(f"  Merged rows   : {len(df):,}")


# %% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 3 · Filter Top Genres
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n▶ Step 2 · Filtering top genres…")

genre_counter = Counter()
for gl in df["genres"]:
    genre_counter.update(gl)

top_genres = [g for g, _ in genre_counter.most_common(TOP_N_GENRES)]

df["genres"] = df["genres"].apply(lambda gl: [g for g in gl if g in top_genres])
df = df[df["genres"].apply(len) > 0].copy()
df.reset_index(drop=True, inplace=True)

genre_names = sorted(top_genres)

print(f"  Keeping top {TOP_N_GENRES} genres → {len(df):,} movies remain")
print(f"  Genres: {genre_names}")


# %% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 4 · Text Preprocessing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n▶ Step 3 · Preprocessing text…")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\[\[.*?\]\]", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text)
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)

def preprocess_text(text):
    return tokenize_and_lemmatize(clean_text(text))

# Process all plots
t0 = time.time()
total = len(df)
cleaned = []
for i, text in enumerate(df["plot"]):
    cleaned.append(preprocess_text(str(text)))
    if (i + 1) % 5000 == 0:
        print(f"  Preprocessed {i+1:,}/{total:,} texts…")

df["clean_plot"] = cleaned
print(f"  Preprocessed {total:,}/{total:,} texts — done ({time.time()-t0:.1f}s)")

# Save processed data
df_save = df[["wiki_id", "name", "clean_plot", "genres"]].copy()
df_save["genres"] = df_save["genres"].apply(json.dumps)
df_save.to_csv(os.path.join(PROCESSED_DIR, "processed_data.csv"), index=False)
print(f"  Saved → {PROCESSED_DIR}/processed_data.csv")


# %% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 5 · Exploratory Data Analysis (EDA)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n▶ Step 4 · Generating EDA charts…")

# ── 5a. Genre Distribution ──
counter = Counter()
for gl in df["genres"]:
    counter.update(gl)
genres_sorted, counts_sorted = zip(*counter.most_common())

fig, ax = plt.subplots(figsize=(12, 6))
colors = sns.color_palette("viridis", len(genres_sorted))
ax.barh(list(reversed(genres_sorted)), list(reversed(counts_sorted)),
        color=list(reversed(colors)))
ax.set_xlabel("Number of Movies")
ax.set_title("Genre Distribution (Top Genres)", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "genre_distribution.png"), dpi=150)
plt.show()
print("  ✓ Genre distribution chart")

# ── 5b. Summary Length Distribution ──
lengths = df["plot"].str.split().str.len()

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(lengths, bins=80, color="#5E81AC", edgecolor="white", alpha=0.85)
ax.set_xlabel("Number of Words")
ax.set_ylabel("Number of Movies")
ax.set_title("Plot Summary Length Distribution", fontsize=14, fontweight="bold")
ax.axvline(lengths.median(), color="#BF616A", linestyle="--",
           label=f"Median = {int(lengths.median())}")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "summary_length_distribution.png"), dpi=150)
plt.show()
print("  ✓ Summary length chart")

# ── 5c. Genres per Movie ──
n_genres = df["genres"].apply(len)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(n_genres, bins=range(1, n_genres.max() + 2),
        color="#A3BE8C", edgecolor="white", align="left")
ax.set_xlabel("Number of Genres per Movie")
ax.set_ylabel("Count")
ax.set_title("Genres per Movie Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "genres_per_movie.png"), dpi=150)
plt.show()
print("  ✓ Genres per movie chart")


# %% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 6 · Train / Validation / Test Split
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n▶ Step 5 · Splitting data (70/15/15)…")

mlb = MultiLabelBinarizer(classes=genre_names)
Y = mlb.fit_transform(df["genres"])
X_texts = df["clean_plot"].values

# 70% train, 30% temp
X_train_text, X_temp_text, Y_train, Y_temp = train_test_split(
    X_texts, Y, test_size=0.30, random_state=RANDOM_STATE
)
# 50/50 of temp → 15% val, 15% test
X_val_text, X_test_text, Y_val, Y_test = train_test_split(
    X_temp_text, Y_temp, test_size=0.50, random_state=RANDOM_STATE
)

print(f"  Train : {len(X_train_text):,}")
print(f"  Val   : {len(X_val_text):,}")
print(f"  Test  : {len(X_test_text):,}")
print(f"  Labels: {Y_train.shape[1]} genres")

joblib.dump(mlb, os.path.join(MODEL_DIR, "label_binarizer.joblib"))


# %% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 7 · TF-IDF Feature Extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n▶ Step 6 · Building TF-IDF features…")

vectorizer = TfidfVectorizer(
    max_features=50_000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode",
)

X_train = vectorizer.fit_transform(X_train_text)
X_val   = vectorizer.transform(X_val_text)
X_test  = vectorizer.transform(X_test_text)

joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))

print(f"  Vocabulary size : {len(vectorizer.vocabulary_):,}")
print(f"  Train shape     : {X_train.shape}")
print(f"  Val shape       : {X_val.shape}")
print(f"  Test shape      : {X_test.shape}")


# %% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 8 · Train Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n▶ Step 7 · Training models…")
models = {}

# ── 8a. Naive Bayes ──
print("\n  Training Naive Bayes…")
t1 = time.time()
nb_model = OneVsRestClassifier(MultinomialNB(alpha=1.0), n_jobs=-1)
nb_model.fit(X_train, Y_train)
models["Naive Bayes"] = nb_model
print(f"  ✓ Naive Bayes trained in {time.time()-t1:.1f}s")

# ── 8b. Logistic Regression ──
print("\n  Training Logistic Regression…")
t1 = time.time()
lr_model = OneVsRestClassifier(
    LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"), n_jobs=-1
)
lr_model.fit(X_train, Y_train)
models["Logistic Regression"] = lr_model
print(f"  ✓ Logistic Regression trained in {time.time()-t1:.1f}s")

# ── 8c. Linear SVM ──
print("\n  Training Linear SVM…")
t1 = time.time()
svm_base = LinearSVC(max_iter=2000, C=1.0, dual="auto")
svm_calibrated = CalibratedClassifierCV(svm_base, cv=3)
svm_model = OneVsRestClassifier(svm_calibrated, n_jobs=-1)
svm_model.fit(X_train, Y_train)
models["Linear SVM"] = svm_model
print(f"  ✓ Linear SVM trained in {time.time()-t1:.1f}s")


# %% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 9 · Evaluate on Validation Set
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n▶ Step 8 · Evaluating on validation set…")

def evaluate_model(model, X, Y_true, label_names):
    Y_pred = model.predict(X)
    results = {
        "subset_accuracy": float(accuracy_score(Y_true, Y_pred)),
        "hamming_loss":    float(hamming_loss(Y_true, Y_pred)),
        "micro_precision": float(precision_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "micro_recall":    float(recall_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "micro_f1":        float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "macro_recall":    float(recall_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "macro_f1":        float(f1_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "weighted_f1":     float(f1_score(Y_true, Y_pred, average="weighted", zero_division=0)),
    }
    report = classification_report(Y_true, Y_pred, target_names=label_names,
                                   zero_division=0, output_dict=True)
    results["per_class"] = report
    return results

all_val_results = {}
for name, model in models.items():
    res = evaluate_model(model, X_val, Y_val, genre_names)
    all_val_results[name] = res
    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║  {name:^36s}  ║")
    print(f"  ╠══════════════════════════════════════╣")
    print(f"  ║  Subset Accuracy : {res['subset_accuracy']:.4f}             ║")
    print(f"  ║  Hamming Loss    : {res['hamming_loss']:.4f}             ║")
    print(f"  ║  Micro F1        : {res['micro_f1']:.4f}             ║")
    print(f"  ║  Macro F1        : {res['macro_f1']:.4f}             ║")
    print(f"  ║  Weighted F1     : {res['weighted_f1']:.4f}             ║")
    print(f"  ╚══════════════════════════════════════╝")

# Pick best
best_name = max(all_val_results, key=lambda k: all_val_results[k]["macro_f1"])
best_model = models[best_name]
print(f"\n  🏆 Best model (macro F1): {best_name}")


# %% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 10 · Final Evaluation on Test Set
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n▶ Step 9 · Final evaluation on TEST set…")

test_results = evaluate_model(best_model, X_test, Y_test, genre_names)

print(f"\n  ╔══════════════════════════════════════╗")
print(f"  ║  {best_name + ' (TEST)':^36s}  ║")
print(f"  ╠══════════════════════════════════════╣")
print(f"  ║  Subset Accuracy : {test_results['subset_accuracy']:.4f}             ║")
print(f"  ║  Hamming Loss    : {test_results['hamming_loss']:.4f}             ║")
print(f"  ║  Micro F1        : {test_results['micro_f1']:.4f}             ║")
print(f"  ║  Macro F1        : {test_results['macro_f1']:.4f}             ║")
print(f"  ║  Weighted F1     : {test_results['weighted_f1']:.4f}             ║")
print(f"  ╚══════════════════════════════════════╝")


# %% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 11 · Model Comparison Chart
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n▶ Step 10a · Model Comparison chart…")

metrics_to_plot = ["micro_f1", "macro_f1", "weighted_f1", "subset_accuracy"]
model_names = list(all_val_results.keys())
x = np.arange(len(metrics_to_plot))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
bar_colors = ["#5E81AC", "#A3BE8C", "#BF616A"]

for i, name in enumerate(model_names):
    vals = [all_val_results[name][m] for m in metrics_to_plot]
    bars = ax.bar(x + i * width, vals, width, label=name, color=bar_colors[i])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_ylabel("Score")
ax.set_title("Model Comparison (Validation)", fontsize=14, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels([m.replace("_", " ").title() for m in metrics_to_plot])
ax.set_ylim(0, 1.05)
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "model_comparison.png"), dpi=150)
plt.show()
print("  ✓ Model comparison chart saved")


# %% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 12 · Per-Genre F1 Chart
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n▶ Step 10b · Per-Genre F1 chart…")

fig, ax = plt.subplots(figsize=(10, 8))
y = np.arange(len(genre_names))
height = 0.25

for i, (mname, res) in enumerate(all_val_results.items()):
    per_class = res["per_class"]
    f1_vals = [per_class.get(g, {}).get("f1-score", 0) for g in genre_names]
    ax.barh(y + i * height, f1_vals, height, label=mname, color=bar_colors[i])

ax.set_yticks(y + height)
ax.set_yticklabels(genre_names, fontsize=9)
ax.set_xlabel("F1-Score")
ax.set_title("Per-Genre F1-Score by Model", fontsize=14, fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "per_genre_f1.png"), dpi=150)
plt.show()
print("  ✓ Per-genre F1 chart saved")


# %% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 13 · Confusion Matrices
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n▶ Step 10c · Confusion matrices for best model…")

Y_test_pred = best_model.predict(X_test)
cms = multilabel_confusion_matrix(Y_test, Y_test_pred)

n = len(genre_names)
cols = 5
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3))
axes_flat = axes.flatten()

for idx, (cm, gname) in enumerate(zip(cms, genre_names)):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes_flat[idx],
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], cbar=False)
    axes_flat[idx].set_title(gname, fontsize=9, fontweight="bold")
    axes_flat[idx].set_ylabel("True")
    axes_flat[idx].set_xlabel("Pred")

for idx in range(n, len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle(f"Per-Genre Confusion Matrices ({best_name})",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f"confusion_matrices_{best_name.lower().replace(' ', '_')}.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Confusion matrices saved")


# %% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 14 · Save Models & Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n▶ Step 10d · Saving models & metrics…")

# Save all models
for name, model in models.items():
    fname = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, os.path.join(MODEL_DIR, fname))
    print(f"  Model saved → {MODEL_DIR}/{fname}")

# Save best with canonical name
joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.joblib"))
print(f"  Best model saved → {MODEL_DIR}/best_model.joblib")

# Save metrics as JSON
for name, res in all_val_results.items():
    fname = name.lower().replace(" ", "_") + "_val_metrics.json"
    with open(os.path.join(METRIC_DIR, fname), "w") as f:
        json.dump(res, f, indent=2)

with open(os.path.join(METRIC_DIR, "best_model_test_metrics.json"), "w") as f:
    json.dump(test_results, f, indent=2)

summary = {
    "best_model": best_name,
    "validation_results": {
        k: {m: v[m] for m in ["micro_f1", "macro_f1", "weighted_f1", "subset_accuracy"]}
        for k, v in all_val_results.items()
    },
    "test_results": {
        m: test_results[m] for m in ["micro_f1", "macro_f1", "weighted_f1", "subset_accuracy"]
    },
}
with open(os.path.join(METRIC_DIR, "model_comparison_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("  ✓ All metrics saved")


# %% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 15 · Inference Function & Demo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n▶ Step 11 · Demo inference…")

def predict_genres(text, model=best_model, vec=vectorizer, binarizer=mlb, threshold=0.3):
    """Predict genres from a raw plot summary string."""
    clean = preprocess_text(text)
    X = vec.transform([clean])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        preds = (probs >= threshold).astype(int)
        if preds.sum() == 0:
            preds[np.argmax(probs)] = 1
        return list(binarizer.inverse_transform(preds.reshape(1, -1))[0])
    else:
        preds = model.predict(X)
        return list(binarizer.inverse_transform(preds)[0])

# ── Demo samples ──
samples = [
    ("Sci-fi adventure",
     "A group of astronauts travel to a distant planet where they discover "
     "an alien civilization that threatens to destroy Earth. They must use "
     "advanced weapons and clever strategy to survive."),

    ("Romantic comedy",
     "A young woman moves to New York City and falls in love with her "
     "neighbor. Through a series of comedic misunderstandings, they "
     "eventually realize they are meant to be together."),

    ("Crime thriller",
     "A detective investigates a series of grisly murders in a small "
     "town. As the body count rises, he realizes the killer is someone "
     "he knows and must confront his own dark past."),

    ("Animated family film",
     "A talking dog and his friends go on a magical adventure through "
     "an enchanted forest to save their village from an evil sorcerer. "
     "Along the way they learn the true meaning of friendship."),

    ("War drama",
     "During World War II, a platoon of soldiers is sent behind enemy "
     "lines to rescue a captured general. They face impossible odds "
     "but their courage and brotherhood carries them through."),
]

print()
for label, text in samples:
    genres = predict_genres(text)
    print(f"  [{label}]")
    print(f"    \"{text[:80]}…\"")
    print(f"    → {', '.join(genres)}\n")


# %% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 16 · Try Your Own! (Interactive)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# %% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 16 · Try Your Own! (Interactive)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("  🎬  INTERACTIVE MOVIE GENRE PREDICTOR")
print("=" * 60)
print("Type a movie plot and press Enter to see the predicted genres.")
print("Type 'quit' or 'exit' to stop.")
print("-" * 60)

while True:
    try:
        # Prompt for user input in the Spyder console
        user_plot = input("\n📝 Enter movie plot: ")
        
        if user_plot.strip().lower() in ['quit', 'exit']:
            print("Exiting interactive mode...")
            break
            
        if not user_plot.strip():
            continue
            
        # Predict genres
        my_genres = predict_genres(user_plot)
        print(f"🎯 Predicted genres: {', '.join(my_genres)}")
        
    except (EOFError, KeyboardInterrupt):
        print("\nExiting interactive mode...")
        break

print("\n" + "=" * 60)
print("  ✅ Pipeline complete! All outputs in models/ and outputs/")
print("=" * 60)
