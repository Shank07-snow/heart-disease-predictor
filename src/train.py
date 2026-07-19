"""
Model training pipeline for the Heart Disease Risk Predictor.

Trains and compares several classifiers with cross-validation, selects the best
by ROC-AUC, evaluates it on a held-out test set, and persists the fitted
pipeline (scaler + model) plus a metrics report to disk.

Run:
    python -m src.train
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_prep import FEATURE_COLUMNS, get_clean_dataframe

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "heart_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

RANDOM_STATE = 42


def build_candidates() -> dict[str, Pipeline]:
    """Return candidate model pipelines (each includes feature scaling)."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=6,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    GradientBoostingClassifier(random_state=RANDOM_STATE),
                ),
            ]
        ),
    }


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    df = get_clean_dataframe()
    X = df[FEATURE_COLUMNS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # --- Model selection by cross-validated ROC-AUC ---
    candidates = build_candidates()
    cv_results: dict[str, float] = {}
    for name, pipe in candidates.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        cv_results[name] = float(scores.mean())
        print(f"{name:>22}: CV ROC-AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")

    best_name = max(cv_results, key=cv_results.get)
    best_model = candidates[best_name]
    print(f"\nBest model: {best_name}")

    # --- Fit best model on full training set and evaluate on test set ---
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "best_model": best_name,
        "cv_roc_auc": cv_results,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_f1": float(f1_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": FEATURE_COLUMNS,
    }

    print("\n=== Held-out test performance ===")
    print(f"Accuracy : {metrics['test_accuracy']:.4f}")
    print(f"F1       : {metrics['test_f1']:.4f}")
    print(f"ROC-AUC  : {metrics['test_roc_auc']:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=["No disease", "Disease"]))

    # --- Refit on the FULL dataset for the deployed model ---
    best_model.fit(X, y)
    joblib.dump(best_model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved model   -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
