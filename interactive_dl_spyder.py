# -*- coding: utf-8 -*-
"""
Deep Learning Interactive Predictor for Spyder
==============================================
Run this file in Spyder to test the trained BiLSTM and DistilBERT
models interactively using the IPython Console.
"""

# %% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Load Models and Start Interactive Loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import os
import sys

# Prevent Transformers from importing TensorFlow and causing a SegFault on Mac
os.environ["USE_TF"] = "NO"
os.environ["USE_TORCH"] = "YES"

# Ensure we can import from src/
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.dl_inference import (
    load_lstm_artifacts,
    load_bert_artifacts,
    predict_genres_lstm,
    predict_genres_bert
)

print("=" * 60)
print("  🤖 LOADING DEEP LEARNING MODELS...")
print("=" * 60)

# Load BiLSTM
try:
    print("Loading BiLSTM model...")
    lstm_model, lstm_vocab, mlb = load_lstm_artifacts("models")
    print("✓ BiLSTM loaded.")
    has_lstm = True
except Exception as e:
    print(f"⚠ Could not load BiLSTM: {e}")
    has_lstm = False

# Load DistilBERT
try:
    print("\nLoading DistilBERT model...")
    bert_model, bert_tokenizer, _ = load_bert_artifacts("models")
    print("✓ DistilBERT loaded.")
    has_bert = True
except Exception as e:
    print(f"⚠ Could not load DistilBERT: {e}")
    has_bert = False

if not has_lstm and not has_bert:
    print("\n❌ Error: Neither deep learning model could be loaded.")
    print("Make sure you have run 'main_deep_learning.py' to train them first!")
else:
    print("\n" + "=" * 60)
    print("  🎬  INTERACTIVE DL GENRE PREDICTOR")
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
            
            # Predict genres
            if has_lstm:
                lstm_genres = predict_genres_lstm(user_plot, lstm_model, lstm_vocab, mlb, threshold=0.3)
                print(f"🧠 BiLSTM Predicted:     {', '.join(lstm_genres)}")
            
            if has_bert:
                bert_genres = predict_genres_bert(user_plot, bert_model, bert_tokenizer, mlb, threshold=0.3)
                print(f"🤖 DistilBERT Predicted: {', '.join(bert_genres)}")
                
            print("-" * 40)
            
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive mode...")
            break

print("\n" + "=" * 60)
print("  ✅ Done!")
print("=" * 60)
