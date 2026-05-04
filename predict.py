#!/usr/bin/env python3
"""
predict.py — CLI script to predict movie genres from a plot summary.

Usage:
    python predict.py "A hero embarks on an epic quest..."
    python predict.py --file path/to/summary.txt
    python predict.py  (interactive mode)
"""

import argparse
import sys

from src.inference import load_inference_artifacts, predict_genres


def main():
    parser = argparse.ArgumentParser(
        description="Predict movie genres from a plot summary.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py "A detective solves a murder mystery in a small town."
  python predict.py --file my_summary.txt
  python predict.py   # interactive mode
        """,
    )
    parser.add_argument("text", nargs="?", help="Plot summary text (inline)")
    parser.add_argument("--file", "-f", help="Path to a text file containing the plot summary")
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.3,
        help="Decision threshold for genre probabilities (default: 0.3)",
    )
    parser.add_argument(
        "--model-dir", "-m", default="models",
        help="Directory containing saved model artifacts (default: models/)",
    )
    args = parser.parse_args()

    # Load artifacts
    print("Loading model…")
    model, vectorizer, mlb = load_inference_artifacts(args.model_dir)
    print("Model loaded ✓\n")

    # Get text
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    elif args.text:
        text = args.text
    else:
        # Interactive mode
        print("Enter a movie plot summary (press Enter twice to submit):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        text = " ".join(lines)

    if not text:
        print("Error: No text provided.")
        sys.exit(1)

    # Predict
    genres = predict_genres(text, model, vectorizer, mlb, threshold=args.threshold)

    print(f"Plot:   \"{text[:120]}{'…' if len(text) > 120 else ''}\"")
    print(f"Genres: {', '.join(genres)}")


if __name__ == "__main__":
    main()
