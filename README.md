# 🎬 Automatic Movie Genre Classification Using Plot Summaries

An end-to-end NLP pipeline that classifies movies into genres based on their plot summaries, built on the **CMU Movie Summary Corpus**.

---

## 📋 Project Overview

| Item | Detail |
|---|---|
| **Dataset** | [CMU Movie Summary Corpus](http://www.cs.cmu.edu/~ark/personas/) |
| **Task** | Multi-label text classification |
| **Features** | TF-IDF (unigram + bigram, 50k max features) |
| **Models** | Multinomial Naive Bayes, Logistic Regression, Linear SVM |
| **Best Model** | Linear SVM (macro F1 ≈ 0.39, micro F1 ≈ 0.49) |
| **Movies** | ~38,900 (from 42,306 with plot summaries) |
| **Genres** | 20 (top genres from 363 total) |

### Genres Classified

Action · Action/Adventure · Adventure · Animation · Black-and-white · Comedy · Crime Fiction · Drama · Family Film · Horror · Indie · Musical · Mystery · Romance Film · Romantic Comedy · Romantic Drama · Science Fiction · Short Film · Thriller · World Cinema

---

## 🗂 Project Structure

```
MovieSummaries/
├── src/
│   ├── __init__.py              # Package init
│   ├── data_loader.py           # Load & merge TSVs, filter genres
│   ├── preprocessing.py         # Text cleaning, tokenization, lemmatization
│   ├── feature_engineering.py   # TF-IDF vectorizer
│   ├── model_training.py        # Train NB, LR, SVM classifiers
│   ├── evaluation.py            # Metrics, confusion matrices, charts
│   └── inference.py             # Predict genres from new text
├── data/
│   ├── raw/                     # Original dataset (TSV files)
│   └── processed/               # Cleaned CSVs
├── models/                      # Saved .joblib models & vectorizer
├── outputs/
│   ├── figures/                 # EDA & evaluation charts (PNG)
│   └── metrics/                 # Classification reports (JSON)
├── notebooks/                   # Optional Jupyter exploration
├── main.py                      # 🚀 End-to-end pipeline
├── predict.py                   # 🔮 CLI inference script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## ⚡ Quick Start

### 1. Prerequisites

- Python 3.9+
- The CMU Movie Summary Corpus files in this directory:
  - `movie.metadata.tsv`
  - `plot_summaries.txt`

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Full Pipeline

```bash
python main.py
```

This single command will:
1. Load and merge the dataset
2. Filter to top 20 genres
3. Clean and preprocess all plot summaries
4. Generate EDA charts
5. Split into train (70%) / validation (15%) / test (15%)
6. Extract TF-IDF features
7. Train 3 classifiers (Naive Bayes, Logistic Regression, Linear SVM)
8. Evaluate all models and pick the best
9. Run final evaluation on held-out test set
10. Save models, metrics, and charts
11. Demo inference on sample plot summaries

### 4. Predict Genres for a New Movie

```bash
# Inline text
python predict.py "A detective investigates murders in a small town."

# From a file
python predict.py --file my_plot_summary.txt

# Interactive mode
python predict.py
```

---

## 📊 Results

### Model Comparison (Validation Set)

| Model | Micro F1 | Macro F1 | Weighted F1 | Subset Accuracy |
|---|---|---|---|---|
| Naive Bayes | ~0.38 | ~0.24 | ~0.32 | ~0.14 |
| Logistic Regression | ~0.45 | ~0.30 | ~0.40 | ~0.15 |
| **Linear SVM** ⭐ | **~0.49** | **~0.39** | **~0.45** | **~0.16** |

> **Note**: Subset accuracy is low because it requires *all* genre labels to match exactly — this is expected for multi-label classification with 20 genres.

### Demo Predictions

```
"A group of astronauts travel to a distant planet..."
  → Science Fiction, Adventure

"A young woman falls in love with her neighbor..."
  → Comedy, Drama, Romance Film, Romantic Comedy

"A detective investigates grisly murders..."
  → Crime Fiction, Horror, Mystery, Thriller
```

---

## 📈 Generated Charts

After running `main.py`, check the `outputs/figures/` directory:

| Chart | Description |
|---|---|
| `genre_distribution.png` | Bar chart of genre frequencies |
| `summary_length_distribution.png` | Histogram of plot lengths |
| `genres_per_movie.png` | Distribution of genres per movie |
| `model_comparison.png` | Side-by-side model metric comparison |
| `per_genre_f1.png` | Per-genre F1 scores across all models |
| `confusion_matrices_*.png` | Per-genre confusion matrix heatmaps |

---

## 🛠 How It Works

### Data Pipeline

1. **Load**: Read `movie.metadata.tsv` (81,741 movies) and `plot_summaries.txt` (42,306 plots)
2. **Merge**: Inner-join on Wikipedia movie ID → 41,796 usable movies
3. **Filter**: Keep only the top 20 genres (movies with rare-only genres are dropped) → 38,888 movies
4. **Clean**: Lowercase, remove wiki markup, remove special characters
5. **Tokenize**: NLTK word_tokenize → remove stopwords → WordNet lemmatization

### Feature Extraction

- **TF-IDF** with sublinear term frequency (`log(1 + tf)`)
- Unigram + bigram features, max 50,000 features
- Minimum document frequency = 3, maximum = 95%

### Classification

- **Multi-label** setup using `OneVsRestClassifier` (each genre is an independent binary classification)
- **Naive Bayes**: Fast baseline with Laplace smoothing
- **Logistic Regression**: L2 regularization with LBFGS solver
- **Linear SVM**: LinearSVC with probability calibration via 3-fold CV

### Evaluation Metrics

- **Micro F1**: Aggregates across all labels (weighted by support)
- **Macro F1**: Average F1 across genres (treats each genre equally)
- **Weighted F1**: Average F1 weighted by genre support
- **Subset Accuracy**: Exact match of entire label set
- **Hamming Loss**: Fraction of incorrect individual labels

---

## 📁 Saved Artifacts

After running the pipeline:

```
models/
├── best_model.joblib            # Best classifier (Linear SVM)
├── tfidf_vectorizer.joblib      # Fitted TF-IDF vectorizer
├── label_binarizer.joblib       # MultiLabelBinarizer
├── naive_bayes.joblib
├── logistic_regression.joblib
└── linear_svm.joblib

outputs/metrics/
├── model_comparison_summary.json
├── best_model_test_metrics.json
├── naive_bayes_val_metrics.json
├── logistic_regression_val_metrics.json
└── linear_svm_val_metrics.json
```

---

## 📚 Dataset

The [CMU Movie Summary Corpus](http://www.cs.cmu.edu/~ark/personas/) was created by David Bamman, Brendan O'Connor, and Noah Smith. It contains:

- **42,306** movie plot summaries from Wikipedia
- **81,741** movie metadata records from Freebase
- **363** unique genres

> David Bamman, Brendan O'Connor and Noah Smith, "Learning Latent Personas of Film Characters," in: *Proceedings of ACL 2013*.

Data is released under a **Creative Commons Attribution-ShareAlike License**.

---

## 🤝 Contributing

Contributions are welcome! Some ideas for improvement:

- Add deep learning models (LSTM, BERT)
- Tune hyperparameters with grid/random search
- Add word embeddings (Word2Vec, GloVe)
- Build a web interface with Streamlit or Gradio
- Experiment with more genres or hierarchical classification
