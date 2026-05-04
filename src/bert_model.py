"""
bert_model.py — DistilBERT fine-tuning for multi-label movie genre classification.

Architecture:
  DistilBERT (distilbert-base-uncased) → [CLS] pooling → Dropout → Linear → Sigmoid

Uses partial layer freezing (freeze first 4 transformer layers) to speed up
training on CPU. Includes gradient accumulation for effective larger batch sizes.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer


# ──────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────


class BertMovieDataset(Dataset):
    """
    Dataset that tokenizes raw text on-the-fly using DistilBERT tokenizer.
    """

    def __init__(self, texts, labels, tokenizer, max_len: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.FloatTensor(self.labels[idx]),
        }


# ──────────────────────────────────────────────────────────────────
# DistilBERT Classification Model
# ──────────────────────────────────────────────────────────────────


class GenreBERT(nn.Module):
    """
    DistilBERT + classification head for multi-label genre prediction.

    Args:
        num_labels:  Number of genre labels
        dropout:     Dropout rate for classifier head
        freeze_layers: Number of transformer layers to freeze (0-6)
    """

    def __init__(self, num_labels: int = 20, dropout: float = 0.3, freeze_layers: int = 4):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # Freeze first N transformer layers
        if freeze_layers > 0:
            # Freeze embeddings
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            # Freeze specified layers
            for i in range(min(freeze_layers, 6)):
                for param in self.bert.transformer.layer[i].parameters():
                    param.requires_grad = False

        # Classification head
        hidden_size = self.bert.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        # Get [CLS] token representation
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        logits = self.classifier(cls_output)  # (batch, num_labels)
        return logits


# ──────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────


def train_bert(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    num_labels: int,
    max_len: int = 256,
    batch_size: int = 16,
    epochs: int = 3,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    grad_accum_steps: int = 4,
    freeze_layers: int = 4,
    patience: int = 2,
    model_dir: str = "models",
):
    """
    Train the DistilBERT model end-to-end.

    Returns:
        model:     Trained GenreBERT
        tokenizer: DistilBertTokenizer
        history:   Dict with training/validation loss and F1 per epoch
    """
    from sklearn.metrics import f1_score as sklearn_f1

    device = torch.device("cpu")
    print(f"  Device: {device}")

    # Load tokenizer
    print("  Loading DistilBERT tokenizer …")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Create datasets
    print("  Creating datasets …")
    train_ds = BertMovieDataset(train_texts, train_labels, tokenizer, max_len)
    val_ds = BertMovieDataset(val_texts, val_labels, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Build model
    print("  Loading DistilBERT model …")
    model = GenreBERT(
        num_labels=num_labels,
        freeze_layers=freeze_layers,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Frozen params:    {total_params - trainable_params:,}")

    # Loss, optimizer, scheduler
    criterion = nn.BCEWithLogitsLoss()

    # Use different learning rates: lower for BERT, higher for classifier
    bert_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "classifier" not in n]
    classifier_params = [p for n, p in model.named_parameters()
                         if p.requires_grad and "classifier" in n]

    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": lr},
        {"params": classifier_params, "lr": lr * 10},
    ], weight_decay=weight_decay)

    # Linear warmup scheduler
    total_steps = len(train_loader) * epochs // grad_accum_steps
    warmup_steps = int(total_steps * 0.1)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training history
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    best_f1 = 0.0
    patience_counter = 0
    os.makedirs(model_dir, exist_ok=True)
    best_model_dir = os.path.join(model_dir, "bert_model")
    os.makedirs(best_model_dir, exist_ok=True)

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_losses = []
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss = loss / grad_accum_steps  # Scale loss for accumulation
            loss.backward()
            train_losses.append(loss.item() * grad_accum_steps)

            # Gradient accumulation step
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (batch_idx + 1) % 100 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"    Epoch {epoch+1}/{epochs} · "
                    f"Batch {batch_idx+1}/{len(train_loader)} · "
                    f"Loss: {loss.item() * grad_accum_steps:.4f} · "
                    f"LR: {current_lr:.2e}"
                )

        avg_train_loss = np.mean(train_losses)

        # ── Validate ──
        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask)
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

        # Save best model
        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(best_model_dir, "model.pt"))
            tokenizer.save_pretrained(best_model_dir)
            print(f"  ✓ Best model saved (F1={best_f1:.4f}) → {best_model_dir}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⏹ Early stopping at epoch {epoch+1} (patience={patience})")
                break

    # Load best weights
    model.load_state_dict(
        torch.load(os.path.join(best_model_dir, "model.pt"), weights_only=True)
    )
    print(f"  Best validation Macro F1: {best_f1:.4f}")

    return model, tokenizer, history


# ──────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────


def predict_bert(
    model,
    texts,
    tokenizer,
    max_len: int = 256,
    batch_size: int = 32,
    threshold: float = 0.3,
) -> np.ndarray:
    """
    Predict multi-label outputs for a list of texts.

    Returns:
        Binary prediction matrix (n_samples, n_labels)
    """
    device = next(model.parameters()).device
    model.eval()

    # Create a dummy labels array
    dummy_labels = np.zeros((len(texts), 1))
    dataset = BertMovieDataset(texts, dummy_labels, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).int().cpu().numpy()
            all_preds.append(preds)
            all_probs.append(probs.cpu().numpy())

    preds = np.vstack(all_preds)
    probs = np.vstack(all_probs)

    # If nothing passes threshold for a sample, pick top-1
    for i in range(len(preds)):
        if preds[i].sum() == 0:
            preds[i][np.argmax(probs[i])] = 1

    return preds


def predict_bert_proba(
    model,
    texts,
    tokenizer,
    max_len: int = 256,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Return probability scores for each label.

    Returns:
        Probability matrix (n_samples, n_labels)
    """
    device = next(model.parameters()).device
    model.eval()

    dummy_labels = np.zeros((len(texts), 1))
    dataset = BertMovieDataset(texts, dummy_labels, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    return np.vstack(all_probs)
