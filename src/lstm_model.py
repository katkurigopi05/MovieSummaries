"""
lstm_model.py — BiLSTM with GloVe embeddings + Self-Attention for multi-label
                movie genre classification.

Architecture:
  Embedding (GloVe 100d) → BiLSTM (2 layers, 128 hidden) → Self-Attention
  → FC → Dropout → FC → Sigmoid

Designed to run on CPU in ~10-20 minutes for 5 epochs on the CMU corpus.
"""

import json
import os
import re
import urllib.request
import zipfile

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ──────────────────────────────────────────────────────────────────
# Vocabulary & Sequence Utilities
# ──────────────────────────────────────────────────────────────────

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


def build_vocab(texts, max_vocab: int = 50_000) -> dict:
    """
    Build a word→index mapping from training texts.
    Reserves index 0 for <PAD> and 1 for <UNK>.
    """
    from collections import Counter

    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())

    most_common = word_counts.most_common(max_vocab - 2)  # reserve PAD, UNK
    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for word, _ in most_common:
        vocab[word] = len(vocab)

    print(f"  Vocabulary size: {len(vocab):,}")
    return vocab


def save_vocab(vocab: dict, path: str):
    """Save vocabulary to a JSON file."""
    with open(path, "w") as f:
        json.dump(vocab, f)
    print(f"  Vocabulary saved → {path}")


def load_vocab(path: str) -> dict:
    """Load vocabulary from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def texts_to_sequences(texts, vocab: dict, max_len: int = 500) -> np.ndarray:
    """
    Convert texts to padded integer sequences.
    Unknown words map to UNK_IDX, sequences are truncated/padded to max_len.
    """
    sequences = []
    for text in texts:
        words = text.split()[:max_len]
        seq = [vocab.get(w, UNK_IDX) for w in words]
        # Pad to max_len
        seq += [PAD_IDX] * (max_len - len(seq))
        sequences.append(seq)
    return np.array(sequences, dtype=np.int64)


# ──────────────────────────────────────────────────────────────────
# GloVe Embeddings
# ──────────────────────────────────────────────────────────────────

GLOVE_DIR = "data/glove"
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_FILE = "glove.6B.100d.txt"


def _download_glove():
    """Download GloVe embeddings if not present."""
    os.makedirs(GLOVE_DIR, exist_ok=True)
    glove_path = os.path.join(GLOVE_DIR, GLOVE_FILE)

    if os.path.exists(glove_path):
        return glove_path

    zip_path = os.path.join(GLOVE_DIR, "glove.6B.zip")

    if not os.path.exists(zip_path):
        print("  Downloading GloVe embeddings (862 MB) — this may take a while …")
        # SSL workaround for macOS
        import ssl
        try:
            ctx = ssl._create_unverified_context()
        except AttributeError:
            ctx = None

        if ctx:
            urllib.request.urlretrieve(GLOVE_URL, zip_path,
                                       reporthook=_progress_hook)
        else:
            urllib.request.urlretrieve(GLOVE_URL, zip_path,
                                       reporthook=_progress_hook)
        print()

    # Extract only the 100d file
    print("  Extracting glove.6B.100d.txt …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract(GLOVE_FILE, GLOVE_DIR)

    return glove_path


def _progress_hook(block_num, block_size, total_size):
    """Download progress indicator."""
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
    mb = downloaded / (1024 * 1024)
    total_mb = total_size / (1024 * 1024) if total_size > 0 else 0
    print(f"\r  Progress: {pct}% ({mb:.0f}/{total_mb:.0f} MB)", end="", flush=True)


def load_glove_embeddings(vocab: dict, embed_dim: int = 100) -> np.ndarray:
    """
    Load GloVe embeddings and create an embedding matrix aligned to vocab.
    Words not found in GloVe are initialized randomly.
    """
    glove_path = _download_glove()
    vocab_size = len(vocab)

    # Initialize with random small values
    embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, embed_dim))
    embedding_matrix[PAD_IDX] = np.zeros(embed_dim)

    # Load GloVe vectors
    print(f"  Loading GloVe embeddings from {glove_path} …")
    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab:
                vec = np.array(parts[1:], dtype=np.float32)
                embedding_matrix[vocab[word]] = vec
                found += 1

    coverage = found / (vocab_size - 2) * 100  # exclude PAD, UNK
    print(f"  GloVe coverage: {found:,}/{vocab_size - 2:,} words ({coverage:.1f}%)")
    return embedding_matrix.astype(np.float32)


# ──────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────


class MovieGenreDataset(Dataset):
    """Simple dataset for integer-encoded text + multi-hot labels."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ──────────────────────────────────────────────────────────────────
# BiLSTM + Self-Attention Model
# ──────────────────────────────────────────────────────────────────


class SelfAttention(nn.Module):
    """Additive self-attention to weight LSTM hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, lstm_output, mask=None):
        # lstm_output: (batch, seq_len, hidden_dim)
        attn_weights = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_weights, dim=1)
        # Weighted sum
        weighted = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return weighted, attn_weights


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM with self-attention for multi-label classification.

    Args:
        vocab_size:   Number of words in vocabulary
        embed_dim:    Embedding dimension (100 for GloVe)
        hidden_dim:   LSTM hidden size per direction
        num_labels:   Number of output labels (genres)
        num_layers:   Number of LSTM layers
        dropout:      Dropout rate
        pretrained_embeddings: Optional pre-trained embedding matrix
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        num_labels: int = 20,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained_embeddings=None,
    ):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(
                torch.FloatTensor(pretrained_embeddings)
            )
            self.embedding.weight.requires_grad = True  # fine-tune embeddings

        # BiLSTM
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Self-attention
        self.attention = SelfAttention(hidden_dim * 2)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x):
        # x: (batch, seq_len)
        mask = (x != PAD_IDX).float()  # (batch, seq_len)

        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden*2)

        # Attention-weighted representation
        attended, _ = self.attention(lstm_out, mask)  # (batch, hidden*2)

        logits = self.classifier(attended)  # (batch, num_labels)
        return logits


# ──────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────


def train_lstm(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    num_labels: int,
    vocab: dict = None,
    max_len: int = 500,
    embed_dim: int = 100,
    hidden_dim: int = 128,
    batch_size: int = 64,
    epochs: int = 5,
    lr: float = 1e-3,
    patience: int = 2,
    model_dir: str = "models",
    use_glove: bool = True,
):
    """
    Train the BiLSTM model end-to-end.

    Returns:
        model:   Trained BiLSTMClassifier
        vocab:   Word→index vocabulary
        history: Dict with training/validation loss and F1 per epoch
    """
    from sklearn.metrics import f1_score as sklearn_f1

    device = torch.device("cpu")
    print(f"  Device: {device}")

    # Build vocabulary if not provided
    if vocab is None:
        vocab = build_vocab(train_texts, max_vocab=50_000)

    # Convert texts to sequences
    print("  Converting texts to sequences …")
    train_seqs = texts_to_sequences(train_texts, vocab, max_len)
    val_seqs = texts_to_sequences(val_texts, vocab, max_len)

    # Create datasets and loaders
    train_ds = MovieGenreDataset(train_seqs, train_labels)
    val_ds = MovieGenreDataset(val_seqs, val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Load GloVe embeddings
    pretrained = None
    if use_glove:
        pretrained = load_glove_embeddings(vocab, embed_dim)

    # Build model
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_labels=num_labels,
        pretrained_embeddings=pretrained,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {param_count:,}")

    # Loss, optimizer, scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    # Training history
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    best_f1 = 0.0
    patience_counter = 0
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "lstm_model.pt")

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_losses = []
        for batch_idx, (seqs, labels) in enumerate(train_loader):
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"    Epoch {epoch+1}/{epochs} · "
                    f"Batch {batch_idx+1}/{len(train_loader)} · "
                    f"Loss: {loss.item():.4f}"
                )

        avg_train_loss = np.mean(train_losses)

        # ── Validate ──
        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                logits = model(seqs)
                loss = criterion(logits, labels)
                val_losses.append(loss.item())

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).int().cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        val_macro_f1 = float(sklearn_f1(all_labels, all_preds, average="macro", zero_division=0))

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_macro_f1"].append(val_macro_f1)

        print(
            f"  Epoch {epoch+1}/{epochs} — "
            f"Train Loss: {avg_train_loss:.4f} · "
            f"Val Loss: {avg_val_loss:.4f} · "
            f"Val Macro F1: {val_macro_f1:.4f}"
        )

        scheduler.step(val_macro_f1)

        # Early stopping
        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved (F1={best_f1:.4f}) → {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⏹ Early stopping at epoch {epoch+1} (patience={patience})")
                break

    # Load best weights
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    print(f"  Best validation Macro F1: {best_f1:.4f}")

    # Save vocabulary
    save_vocab(vocab, os.path.join(model_dir, "lstm_vocab.json"))

    return model, vocab, history


# ──────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────


def predict_lstm(
    model,
    texts,
    vocab: dict,
    max_len: int = 500,
    threshold: float = 0.3,
) -> np.ndarray:
    """
    Predict multi-label outputs for a list of texts.

    Returns:
        Binary prediction matrix (n_samples, n_labels)
    """
    device = next(model.parameters()).device
    model.eval()

    sequences = texts_to_sequences(texts, vocab, max_len)
    dataset = MovieGenreDataset(sequences, np.zeros((len(texts), 1)))  # dummy labels
    loader = DataLoader(dataset, batch_size=64)

    all_preds = []
    with torch.no_grad():
        for seqs, _ in loader:
            seqs = seqs.to(device)
            logits = model(seqs)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).int().cpu().numpy()
            all_preds.append(preds)

    preds = np.vstack(all_preds)

    # If nothing passes threshold for a sample, pick top-1
    for i in range(len(preds)):
        if preds[i].sum() == 0:
            probs_i = torch.sigmoid(model(torch.LongTensor(sequences[i:i+1]).to(device)))
            preds[i][probs_i.argmax().item()] = 1

    return preds


def predict_lstm_proba(
    model,
    texts,
    vocab: dict,
    max_len: int = 500,
) -> np.ndarray:
    """
    Return probability scores for each label.

    Returns:
        Probability matrix (n_samples, n_labels)
    """
    device = next(model.parameters()).device
    model.eval()

    sequences = texts_to_sequences(texts, vocab, max_len)
    dataset = MovieGenreDataset(sequences, np.zeros((len(texts), 1)))
    loader = DataLoader(dataset, batch_size=64)

    all_probs = []
    with torch.no_grad():
        for seqs, _ in loader:
            seqs = seqs.to(device)
            logits = model(seqs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    return np.vstack(all_probs)
