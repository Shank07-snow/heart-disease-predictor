"""Unit tests for the heart disease prediction pipeline."""
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.data_prep import FEATURE_COLUMNS, get_clean_dataframe

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "heart_model.joblib"


def test_clean_data_has_no_missing_values():
    df = get_clean_dataframe()
    assert df.isna().sum().sum() == 0


def test_target_is_binary():
    df = get_clean_dataframe()
    assert set(df["target"].unique()).issubset({0, 1})


def test_expected_feature_count():
    df = get_clean_dataframe()
    assert len(FEATURE_COLUMNS) == 13
    assert all(col in df.columns for col in FEATURE_COLUMNS)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained yet")
def test_model_predicts_probability_in_range():
    model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame(
        [[54, 1, 4, 130, 245, 0, 0, 150, 0, 1.0, 2, 0, 3]],
        columns=FEATURE_COLUMNS,
    )
    prob = model.predict_proba(sample)[0, 1]
    assert 0.0 <= prob <= 1.0
