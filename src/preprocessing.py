"""
preprocessing.py — Text cleaning and preprocessing for plot summaries.

Pipeline:
  1. Lowercase
  2. Remove HTML / wiki markup
  3. Remove special characters (keep letters, digits, spaces)
  4. Tokenize
  5. Remove stopwords
  6. Lemmatize
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# Ensure NLTK data is available (silent if already downloaded)
# ---------------------------------------------------------------------------
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

for resource in ["punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, strip markup and special chars."""
    text = text.lower()
    # Remove wiki-style links  [[...]]
    text = re.sub(r"\[\[.*?\]\]", " ", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove everything except letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> str:
    """Tokenize, remove stopwords, and lemmatize. Returns joined string."""
    tokens = word_tokenize(text)
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOP_WORDS and len(tok) > 2
    ]
    return " ".join(tokens)


def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline for a single text string."""
    text = clean_text(text)
    text = tokenize_and_lemmatize(text)
    return text


def preprocess_dataframe(df, text_col: str = "plot", new_col: str = "clean_plot"):
    """
    Apply full preprocessing to a DataFrame column.
    Adds a new column with cleaned text.
    Prints progress every 5,000 rows.
    """
    import pandas as pd

    total = len(df)
    cleaned = []
    for i, text in enumerate(df[text_col]):
        cleaned.append(preprocess_text(str(text)))
        if (i + 1) % 5000 == 0:
            print(f"  Preprocessed {i + 1:,}/{total:,} texts …")
    df = df.copy()
    df[new_col] = cleaned
    print(f"  Preprocessed {total:,}/{total:,} texts — done.")
    return df


if __name__ == "__main__":
    sample = "The <b>hero</b> goes on an ADVENTURE [[link]] to save the world! http://example.com"
    print("Original:", sample)
    print("Cleaned: ", preprocess_text(sample))
