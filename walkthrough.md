# Walkthrough: Deep Learning for Movie Genre Classification

## Summary

Added two deep learning models — **BiLSTM** (with GloVe embeddings + self-attention) and **DistilBERT** (fine-tuned) — to the existing ML pipeline. Both models significantly outperform the previous best (Linear SVM).

---

## Results: 5-Model Comparison

| Model | Macro F1 | Micro F1 | Weighted F1 | Training Time |
|-------|----------|----------|-------------|---------------|
| Naive Bayes | 0.067 | 0.284 | 0.184 | ~2s |
| Logistic Regression | 0.297 | 0.446 | 0.396 | ~30s |
| Linear SVM | 0.386 | 0.491 | 0.452 | ~120s |
| **BiLSTM** | **0.441** | **0.535** | **0.510** | ~19 min |
| **DistilBERT** 🏆 | **0.521** | **0.589** | **0.574** | ~4.3 hrs |

> [!IMPORTANT]
> **DistilBERT is the new best model** with a **35% improvement** in Macro F1 over Linear SVM (0.521 vs 0.386). BiLSTM also beats all ML baselines.

### 5-Model Comparison Chart
![All Models Comparison](/Users/gopikrishnareddykatkuri/.gemini/antigravity/brain/a67ecca7-d176-4778-a342-db89e1a8e788/all_models_comparison.png)

---

## Training Curves

### BiLSTM (5 epochs)
- Trained all 5 epochs, F1 improved steadily from 0.04 → 0.36
- Val loss showed slight overfitting by epoch 4-5 (train loss continuing to decrease while val loss plateaued)

![BiLSTM Training Curves](/Users/gopikrishnareddykatkuri/.gemini/antigravity/brain/a67ecca7-d176-4778-a342-db89e1a8e788/lstm_training_curves.png)

### DistilBERT (3 epochs)
- Steady improvement across all 3 epochs
- F1 improved from 0.34 → 0.42 (validation)
- Still improving — more epochs would likely help further

![DistilBERT Training Curves](/Users/gopikrishnareddykatkuri/.gemini/antigravity/brain/a67ecca7-d176-4778-a342-db89e1a8e788/bert_training_curves.png)

---

## Demo Inference Results

| Sample Plot | BiLSTM | DistilBERT |
|-------------|--------|------------|
| Astronauts discover alien civilization… | Action, Horror, Sci-Fi | Action, Adventure, Sci-Fi |
| Young woman falls in love in NYC… | Black-and-white, Comedy, Romance, Romantic comedy | Comedy, Drama, Romance, Romantic comedy |
| Detective investigates grisly murders… | Horror, Thriller | Crime Fiction, Drama, Horror, Mystery, Thriller |

> [!TIP]
> DistilBERT predictions are notably more specific and nuanced — e.g., it correctly identifies "Crime Fiction" and "Mystery" for the detective story.

---

## Architecture Details

### BiLSTM
- **Embeddings**: GloVe 100d (87.9% vocab coverage, 50K words)
- **Encoder**: 2-layer BiLSTM (hidden=128) + Self-Attention
- **Classifier**: FC(256→128) → ReLU → Dropout → FC(128→20) → Sigmoid
- **Parameters**: 5.7M trainable
- **Loss**: BCEWithLogitsLoss

### DistilBERT
- **Backbone**: `distilbert-base-uncased` (66M params, 4 layers frozen)
- **Trainable**: 14.4M params (top 2 transformer layers + classifier head)
- **Classifier**: Dropout → FC(768→256) → ReLU → FC(256→20) → Sigmoid
- **Optimizer**: AdamW with differential LR (2e-5 for BERT, 2e-4 for classifier)
- **Scheduler**: Linear warmup (10%) + linear decay

---

## Files Created/Modified

### New Files
| File | Purpose |
|------|---------|
| [lstm_model.py](file:///Users/gopikrishnareddykatkuri/Downloads/MovieSummaries/src/lstm_model.py) | BiLSTM model, GloVe loading, vocab, training loop |
| [bert_model.py](file:///Users/gopikrishnareddykatkuri/Downloads/MovieSummaries/src/bert_model.py) | DistilBERT model, tokenization, fine-tuning loop |
| [dl_evaluation.py](file:///Users/gopikrishnareddykatkuri/Downloads/MovieSummaries/src/dl_evaluation.py) | DL evaluation, training curves, 5-model comparison charts |
| [dl_inference.py](file:///Users/gopikrishnareddykatkuri/Downloads/MovieSummaries/src/dl_inference.py) | Unified DL inference (load models, predict genres) |
| [main_deep_learning.py](file:///Users/gopikrishnareddykatkuri/Downloads/MovieSummaries/main_deep_learning.py) | End-to-end DL pipeline (10 steps) |
| [predict_dl.py](file:///Users/gopikrishnareddykatkuri/Downloads/MovieSummaries/predict_dl.py) | CLI tool for DL inference |

### Modified Files
| File | Change |
|------|--------|
| [requirements.txt](file:///Users/gopikrishnareddykatkuri/Downloads/MovieSummaries/requirements.txt) | Added `torch`, `transformers` |

### Generated Artifacts
| File | Description |
|------|-------------|
| `models/lstm_model.pt` | Saved LSTM model weights |
| `models/lstm_vocab.json` | LSTM vocabulary (50K words) |
| `models/bert_model/model.pt` | Saved DistilBERT weights |
| `models/bert_model/tokenizer_config.json` | BERT tokenizer config |
| `outputs/figures/all_models_comparison.png` | 5-model comparison chart |
| `outputs/figures/lstm_training_curves.png` | LSTM loss/F1 curves |
| `outputs/figures/bert_training_curves.png` | BERT loss/F1 curves |
| `outputs/figures/confusion_matrices_bilstm.png` | BiLSTM confusion matrices |
| `outputs/figures/confusion_matrices_distilbert.png` | DistilBERT confusion matrices |
| `outputs/metrics/lstm_test_metrics.json` | LSTM test metrics |
| `outputs/metrics/bert_test_metrics.json` | BERT test metrics |
| `outputs/metrics/dl_comparison_summary.json` | DL summary |

---

## How to Use

### Run the full DL pipeline
```bash
python3 main_deep_learning.py
```

### Predict genres with CLI
```bash
python3 predict_dl.py --model both --text "A spy infiltrates an enemy organization..."
python3 predict_dl.py --model bert --text "Two friends open a bakery..."
python3 predict_dl.py --model lstm --text "A zombie apocalypse threatens humanity..."
```

## Verification
- ✅ `main_deep_learning.py` ran end-to-end (exit code 0)
- ✅ All model artifacts saved to `models/`
- ✅ All charts and metrics saved to `outputs/`
- ✅ Demo inference produced sensible genre predictions
- ✅ 5-model comparison chart generated successfully
