#!/usr/bin/env python3
"""
predict_dl.py — CLI tool for movie genre prediction using deep learning models.

Usage:
    python predict_dl.py --model lstm --text "A spaceship crew discovers..."
    python predict_dl.py --model bert --text "A young couple falls in love..."
    python predict_dl.py --model both --text "A detective investigates murders..."
"""

import argparse
import sys
import os

# Prevent Transformers from importing TensorFlow and causing a SegFault on Mac
os.environ["USE_TF"] = "NO"
os.environ["USE_TORCH"] = "YES"


def main():
    parser = argparse.ArgumentParser(
        description="Predict movie genres from a plot summary using deep learning models."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lstm", "bert", "both"],
        default="both",
        help="Which model to use: lstm, bert, or both (default: both)",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Plot summary text to classify",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Classification threshold (default: 0.3)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing saved models (default: models)",
    )

    args = parser.parse_args()

    print("━" * 60)
    print("  🎬 Movie Genre Prediction (Deep Learning)")
    print("━" * 60)
    print(f"\n  Text: \"{args.text[:100]}{'…' if len(args.text) > 100 else ''}\"")
    print(f"  Threshold: {args.threshold}")
    print()

    if args.model in ("lstm", "both"):
        try:
            from src.dl_inference import load_lstm_artifacts, predict_genres_lstm

            print("  Loading BiLSTM model …")
            model, vocab, mlb = load_lstm_artifacts(args.model_dir)
            genres = predict_genres_lstm(args.text, model, vocab, mlb, threshold=args.threshold)
            print(f"  🧠 BiLSTM → {genres}")
        except FileNotFoundError as e:
            print(f"  ⚠ BiLSTM model not found: {e}")
            print("    Run 'python main_deep_learning.py' first to train the model.")
        except Exception as e:
            print(f"  ✗ BiLSTM error: {e}")

    if args.model in ("bert", "both"):
        try:
            from src.dl_inference import load_bert_artifacts, predict_genres_bert

            print("  Loading DistilBERT model …")
            model, tokenizer, mlb = load_bert_artifacts(args.model_dir)
            genres = predict_genres_bert(args.text, model, tokenizer, mlb, threshold=args.threshold)
            print(f"  🤖 DistilBERT → {genres}")
        except FileNotFoundError as e:
            print(f"  ⚠ DistilBERT model not found: {e}")
            print("    Run 'python main_deep_learning.py' first to train the model.")
        except Exception as e:
            print(f"  ✗ DistilBERT error: {e}")

    print()


if __name__ == "__main__":
    main()
