"""
model_training.py — Train multi-label genre classifiers.

Models:
  1. Multinomial Naive Bayes
  2. Logistic Regression
  3. Linear SVM (via LinearSVC + CalibratedClassifierCV)
"""

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def train_naive_bayes(X_train, Y_train):
    """Train a OneVsRest Multinomial Naive Bayes classifier."""
    model = OneVsRestClassifier(MultinomialNB(alpha=1.0), n_jobs=-1)
    model.fit(X_train, Y_train)
    return model


def train_logistic_regression(X_train, Y_train):
    """Train a OneVsRest Logistic Regression classifier."""
    model = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
        n_jobs=-1,
    )
    model.fit(X_train, Y_train)
    return model


def train_linear_svm(X_train, Y_train):
    """Train a OneVsRest Linear SVM (with probability calibration)."""
    base_svm = LinearSVC(max_iter=2000, C=1.0, dual="auto")
    calibrated = CalibratedClassifierCV(base_svm, cv=3)
    model = OneVsRestClassifier(calibrated, n_jobs=-1)
    model.fit(X_train, Y_train)
    return model


def save_model(model, path: str):
    """Persist a trained model to disk."""
    joblib.dump(model, path)
    print(f"  Model saved → {path}")


def load_model(path: str):
    """Load a trained model from disk."""
    return joblib.load(path)
