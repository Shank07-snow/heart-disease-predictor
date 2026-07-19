"""
Data preparation for the Heart Disease Risk Predictor.

Loads the raw UCI Cleveland Heart Disease dataset, cleans it, assigns proper
column names, handles missing values, and binarises the target
(0 = no disease, 1 = disease present).

Source: UCI Machine Learning Repository - Heart Disease Data Set
https://archive.ics.uci.edu/dataset/45/heart+disease
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# 13 clinical features + the diagnosis target, in the raw column order.
COLUMN_NAMES = [
    "age",        # age in years
    "sex",        # 1 = male, 0 = female
    "cp",         # chest pain type (1-4)
    "trestbps",   # resting blood pressure (mm Hg)
    "chol",       # serum cholesterol (mg/dl)
    "fbs",        # fasting blood sugar > 120 mg/dl (1 = true)
    "restecg",    # resting ECG results (0-2)
    "thalach",    # maximum heart rate achieved
    "exang",      # exercise-induced angina (1 = yes)
    "oldpeak",    # ST depression induced by exercise
    "slope",      # slope of the peak exercise ST segment (1-3)
    "ca",         # number of major vessels (0-3) coloured by fluoroscopy
    "thal",       # 3 = normal, 6 = fixed defect, 7 = reversible defect
    "target",     # diagnosis of heart disease (0 = absent, 1-4 = present)
]

FEATURE_COLUMNS = COLUMN_NAMES[:-1]

RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed.cleveland.data"
CLEAN_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "heart_clean.csv"


def load_raw(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw comma-separated Cleveland data with proper headers."""
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe.

    - Impute the small number of missing values in `ca` and `thal` with the
      column median (both are low-cardinality clinical fields).
    - Binarise the target: any value > 0 means disease is present.
    """
    df = df.copy()

    # Impute missing values (only `ca` and `thal` have them in Cleveland data).
    for col in ["ca", "thal"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Binarise diagnosis: 0 = no disease, 1-4 -> 1 (disease present).
    df["target"] = (df["target"] > 0).astype(int)

    return df


def get_clean_dataframe() -> pd.DataFrame:
    """Convenience helper: load + clean in one call."""
    return clean(load_raw())


def main() -> None:
    df = get_clean_dataframe()
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Saved cleaned dataset -> {CLEAN_DATA_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Disease prevalence: {df['target'].mean():.1%}")
    print(df.head())


if __name__ == "__main__":
    main()
