# -*- coding: utf-8 -*-
"""
Movie Genre Classification — Full Pipeline Spyder Version
=========================================================
Automatic Movie Genre Classification Using Plot Summaries
Includes: TF-IDF (NB, LR, SVM) + GloVe BiLSTM + HuggingFace DistilBERT

Run cell-by-cell in Spyder (Ctrl+Enter) or run the entire file.
Each section is marked with '# %%' for Spyder's cell navigation.

NOTE: Deep Learning training (BiLSTM and DistilBERT) can take 
significant time on CPU (10-40+ mins).
"""

# %% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 1 · Imports & Setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import os
import re
import ssl
import json
import time
import urllib.request
import zipfile
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
    accuracy_score, classification_report, f1_score,
    hamming_loss, multilabel_confusion_matrix, precision_score, recall_score
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer

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
X_raw_texts = df["plot"].values # Raw texts for BERT

# 70% train, 30% temp
X_train_text, X_temp_text, Y_train, Y_temp, X_train_raw, X_temp_raw = train_test_split(
    X_texts, Y, X_raw_texts, test_size=0.30, random_state=RANDOM_STATE
)
# 50/50 of temp → 15% val, 15% test
X_val_text, X_test_text, Y_val, Y_test, X_val_raw, X_test_raw = train_test_split(
    X_temp_text, Y_temp, X_temp_raw, test_size=0.50, random_state=RANDOM_STATE
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
#  CELL 10 · Deep Learning Architectures (BiLSTM & DistilBERT)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n▶ Step 8.5 · Defining Deep Learning PyTorch Classes…")

# ──────────────────────────────────────────────────────────────────
# BiLSTM CLASSES & UTILS
# ──────────────────────────────────────────────────────────────────
PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"
PAD_IDX, UNK_IDX = 0, 1

def build_vocab(texts, max_vocab=50000):
    word_counts = Counter()
    for text in texts: word_counts.update(text.split())
    most_common = word_counts.most_common(max_vocab - 2)
    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for word, _ in most_common: vocab[word] = len(vocab)
    return vocab

def texts_to_sequences(texts, vocab, max_len=500):
    sequences = []
    for text in texts:
        seq = [vocab.get(w, UNK_IDX) for w in text.split()[:max_len]]
        seq += [PAD_IDX] * (max_len - len(seq))
        sequences.append(seq)
    return np.array(sequences, dtype=np.int64)

class MovieGenreDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.labels[idx]

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.Tanh(), nn.Linear(hidden_dim//2, 1))
    def forward(self, lstm_output, mask=None):
        attn_weights = self.attention(lstm_output).squeeze(-1)
        if mask is not None: attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_weights, dim=1)
        weighted = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return weighted, attn_weights

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_labels=20, num_layers=2, dropout=0.3, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(torch.FloatTensor(pretrained_embeddings))
            self.embedding.weight.requires_grad = True
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.attention = SelfAttention(hidden_dim * 2)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_labels))
    def forward(self, x):
        mask = (x != PAD_IDX).float()
        lstm_out, _ = self.lstm(self.embedding(x))
        attended, _ = self.attention(lstm_out, mask)
        return self.classifier(attended)

def load_glove_embeddings(vocab, embed_dim=100):
    glove_dir = "data/glove"
    glove_file = "glove.6B.100d.txt"
    os.makedirs(glove_dir, exist_ok=True)
    glove_path = os.path.join(glove_dir, glove_file)
    if not os.path.exists(glove_path):
        print("  Downloading GloVe embeddings (862 MB)…")
        zip_path = os.path.join(glove_dir, "glove.6B.zip")
        urllib.request.urlretrieve("https://nlp.stanford.edu/data/glove.6B.zip", zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extract(glove_file, glove_dir)
    
    embedding_matrix = np.random.uniform(-0.25, 0.25, (len(vocab), embed_dim))
    embedding_matrix[PAD_IDX] = np.zeros(embed_dim)
    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if parts[0] in vocab:
                embedding_matrix[vocab[parts[0]]] = np.array(parts[1:], dtype=np.float32)
                found += 1
    print(f"  GloVe coverage: {found:,}/{len(vocab)-2:,} words")
    return embedding_matrix.astype(np.float32)

# ──────────────────────────────────────────────────────────────────
# BERT CLASSES & UTILS
# ──────────────────────────────────────────────────────────────────

class BertMovieDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts, self.labels, self.tokenizer, self.max_len = texts, labels, tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(str(self.texts[idx]), add_special_tokens=True, max_length=self.max_len, padding="max_length", truncation=True, return_attention_mask=True, return_tensors="pt")
        return {"input_ids": encoding["input_ids"].squeeze(0), "attention_mask": encoding["attention_mask"].squeeze(0), "labels": torch.FloatTensor(self.labels[idx])}

class GenreBERT(nn.Module):
    def __init__(self, num_labels=20, dropout=0.3, freeze_layers=4):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        if freeze_layers > 0:
            for p in self.bert.embeddings.parameters(): p.requires_grad = False
            for i in range(min(freeze_layers, 6)):
                for p in self.bert.transformer.layer[i].parameters(): p.requires_grad = False
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(768, 256), nn.ReLU(), nn.Dropout(dropout/2), nn.Linear(256, num_labels))
    def forward(self, input_ids, attention_mask):
        cls_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

print("  ✓ Deep Learning Classes defined.")


# %% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 11 · Train BiLSTM Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n▶ Step 9 · Training BiLSTM Model…")

import torch.optim as optim
from sklearn.metrics import f1_score as sklearn_f1

device = torch.device("cpu")
lstm_vocab = build_vocab(X_train_text, max_vocab=50000)
with open(os.path.join(MODEL_DIR, "lstm_vocab.json"), "w") as f:
    json.dump(lstm_vocab, f)

train_seqs = texts_to_sequences(X_train_text, lstm_vocab, 500)
val_seqs = texts_to_sequences(X_val_text, lstm_vocab, 500)

train_loader = DataLoader(MovieGenreDataset(train_seqs, Y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(MovieGenreDataset(val_seqs, Y_val), batch_size=64)

pretrained_embeds = load_glove_embeddings(lstm_vocab, 100)
lstm_model = BiLSTMClassifier(len(lstm_vocab), pretrained_embeddings=pretrained_embeds).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

lstm_epochs = 3 # reduced for single-script execution
best_lstm_f1 = 0.0
best_lstm_path = os.path.join(MODEL_DIR, "lstm_model.pt")

t_start = time.time()
for epoch in range(lstm_epochs):
    lstm_model.train()
    for seqs, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(lstm_model(seqs.to(device)), labels.to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
        optimizer.step()
        
    lstm_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, labels in val_loader:
            probs = torch.sigmoid(lstm_model(seqs.to(device)))
            all_preds.append((probs >= 0.5).int().cpu().numpy())
            all_labels.append(labels.numpy())
            
    val_f1 = float(sklearn_f1(np.vstack(all_labels), np.vstack(all_preds), average="macro", zero_division=0))
    print(f"  Epoch {epoch+1}/{lstm_epochs} · Val Macro F1: {val_f1:.4f}")
    
    if val_f1 > best_lstm_f1:
        best_lstm_f1 = val_f1
        torch.save(lstm_model.state_dict(), best_lstm_path)
    scheduler.step(val_f1)

lstm_model.load_state_dict(torch.load(best_lstm_path, weights_only=True))
print(f"  ✓ BiLSTM trained in {time.time()-t_start:.1f}s (Best F1: {best_lstm_f1:.4f})")
models["BiLSTM"] = lstm_model # Add to our models dictionary


# %% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 12 · Train DistilBERT Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n▶ Step 10 · Training DistilBERT Model…")

bert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
train_loader_bert = DataLoader(BertMovieDataset(X_train_raw, Y_train, bert_tokenizer, 256), batch_size=16, shuffle=True)
val_loader_bert = DataLoader(BertMovieDataset(X_val_raw, Y_val, bert_tokenizer, 256), batch_size=16)

bert_model = GenreBERT(freeze_layers=4).to(device)

bert_epochs = 2 # reduced for single-script execution
grad_accum_steps = 4
bert_optimizer = optim.AdamW(bert_model.parameters(), lr=2e-5)
bert_criterion = nn.BCEWithLogitsLoss()

best_bert_f1 = 0.0
bert_dir = os.path.join(MODEL_DIR, "bert_model")
os.makedirs(bert_dir, exist_ok=True)
best_bert_path = os.path.join(bert_dir, "model.pt")

t_start = time.time()
for epoch in range(bert_epochs):
    bert_model.train()
    bert_optimizer.zero_grad()
    for batch_idx, batch in enumerate(train_loader_bert):
        logits = bert_model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        loss = bert_criterion(logits, batch["labels"].to(device)) / grad_accum_steps
        loss.backward()
        if (batch_idx + 1) % grad_accum_steps == 0:
            nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
            bert_optimizer.step()
            bert_optimizer.zero_grad()
            
    bert_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader_bert:
            probs = torch.sigmoid(bert_model(batch["input_ids"].to(device), batch["attention_mask"].to(device)))
            all_preds.append((probs >= 0.5).int().cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
            
    val_f1 = float(sklearn_f1(np.vstack(all_labels), np.vstack(all_preds), average="macro", zero_division=0))
    print(f"  Epoch {epoch+1}/{bert_epochs} · Val Macro F1: {val_f1:.4f}")
    
    if val_f1 > best_bert_f1:
        best_bert_f1 = val_f1
        torch.save(bert_model.state_dict(), best_bert_path)
        bert_tokenizer.save_pretrained(bert_dir)

bert_model.load_state_dict(torch.load(best_bert_path, weights_only=True))
print(f"  ✓ DistilBERT trained in {time.time()-t_start:.1f}s (Best F1: {best_bert_f1:.4f})")
models["DistilBERT"] = bert_model # Add to our models dictionary


# %% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 13 · Final Evaluation of ALL Models on Test Set
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n▶ Step 11 · Evaluating all models on TEST set…")

# Deep Learning inference wrappers
def predict_dl_lstm(model, texts):
    model.eval()
    seqs = texts_to_sequences(texts, lstm_vocab, 500)
    preds = []
    for i in range(0, len(seqs), 64):
        batch = torch.LongTensor(seqs[i:i+64]).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(batch))
            preds.append((probs >= 0.3).int().cpu().numpy()) # Threshold 0.3
    return np.vstack(preds)

def predict_dl_bert(model, texts):
    model.eval()
    preds = []
    for i in range(0, len(texts), 32):
        batch_texts = list(texts[i:i+32])
        enc = bert_tokenizer(batch_texts, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
        with torch.no_grad():
            probs = torch.sigmoid(model(enc["input_ids"].to(device), enc["attention_mask"].to(device)))
            preds.append((probs >= 0.3).int().cpu().numpy()) # Threshold 0.3
    return np.vstack(preds)

# Wrap models for unified evaluation
class WrappedModel:
    def __init__(self, name, model): self.name, self.model = name, model
    def predict(self, texts_or_features):
        if self.name == "BiLSTM": return predict_dl_lstm(self.model, texts_or_features)
        if self.name == "DistilBERT": return predict_dl_bert(self.model, texts_or_features)
        return self.model.predict(texts_or_features)

test_results_all = {}

for name, model in models.items():
    wrapped = WrappedModel(name, model)
    # ML models take X_test (TF-IDF features). DL models take texts.
    if name in ["BiLSTM"]: inputs = X_test_text
    elif name in ["DistilBERT"]: inputs = X_test_raw
    else: inputs = X_test
    
    Y_pred = wrapped.predict(inputs)
    
    # Fallback to top-1 if no genres predicted
    if name in ["BiLSTM", "DistilBERT"]:
        for i in range(len(Y_pred)):
            if Y_pred[i].sum() == 0:
                if name == "BiLSTM":
                    probs = torch.sigmoid(model(torch.LongTensor(texts_to_sequences([X_test_text[i]], lstm_vocab, 500)).to(device)))
                else:
                    enc = bert_tokenizer([X_test_raw[i]], padding="max_length", max_length=256, truncation=True, return_tensors="pt")
                    probs = torch.sigmoid(model(enc["input_ids"].to(device), enc["attention_mask"].to(device)))
                Y_pred[i][probs.argmax().item()] = 1
    
    res = {
        "macro_f1": float(f1_score(Y_test, Y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(Y_test, Y_pred, average="micro", zero_division=0)),
        "subset_accuracy": float(accuracy_score(Y_test, Y_pred))
    }
    test_results_all[name] = res
    print(f"  {name:20s} → Macro F1: {res['macro_f1']:.4f} | Micro F1: {res['micro_f1']:.4f}")

best_name = max(test_results_all, key=lambda k: test_results_all[k]["macro_f1"])
print(f"\n  🏆 Best overall model: {best_name}")

# %% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 14 · Visualizations & Saving
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n▶ Step 12 · Plotting comparison chart…")

fig, ax = plt.subplots(figsize=(10, 6))
names = list(test_results_all.keys())
macro_f1s = [test_results_all[n]["macro_f1"] for n in names]
micro_f1s = [test_results_all[n]["micro_f1"] for n in names]

x = np.arange(len(names))
width = 0.35
ax.bar(x - width/2, macro_f1s, width, label='Macro F1', color='#5E81AC')
ax.bar(x + width/2, micro_f1s, width, label='Micro F1', color='#A3BE8C')

ax.set_ylabel('Score')
ax.set_title('5-Model Comparison on Test Set')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "full_model_comparison.png"))
plt.show()


# %% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CELL 15 · Interactive 5-Model Predictor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("  🎬  INTERACTIVE FULL-PIPELINE PREDICTOR")
print("=" * 60)
print("Paste your movie plot, then press Enter TWICE to submit.")
print("Type 'quit' or 'exit' to stop.")
print("-" * 60)

while True:
    try:
        print("\n📝 Enter movie plot:")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
            
        user_plot = " ".join(lines)
        
        if user_plot.strip().lower() in ['quit', 'exit']:
            print("Exiting interactive mode...")
            break
            
        if not user_plot.strip():
            continue
            
        print("\n" + "-" * 40)
        
        # 1. Prepare ML Inputs
        clean_p = tokenize_and_lemmatize(clean_text(user_plot))
        x_tfidf = vectorizer.transform([clean_p])
        
        # 2. ML Predictions
        for ml_name in ["Naive Bayes", "Logistic Regression", "Linear SVM"]:
            if ml_name in models:
                preds = models[ml_name].predict(x_tfidf)
                genres = mlb.inverse_transform(preds)[0]
                print(f"📊 {ml_name:19s}: {', '.join(genres)}")
                
        # 3. DL Predictions
        if "BiLSTM" in models:
            preds_lstm = WrappedModel("BiLSTM", models["BiLSTM"]).predict([clean_p])
            genres_lstm = mlb.inverse_transform(preds_lstm)[0]
            print(f"🧠 BiLSTM Predicted   : {', '.join(genres_lstm)}")
            
        if "DistilBERT" in models:
            preds_bert = WrappedModel("DistilBERT", models["DistilBERT"]).predict([user_plot]) # Raw text
            genres_bert = mlb.inverse_transform(preds_bert)[0]
            print(f"🤖 DistilBERT Predicted: {', '.join(genres_bert)}")
            
        print("-" * 40)
        
    except (EOFError, KeyboardInterrupt):
        print("\nExiting interactive mode...")
        break

print("\n" + "=" * 60)
print("  ✅ Pipeline complete!")
print("=" * 60)
