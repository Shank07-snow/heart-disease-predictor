"""
Heart Disease Risk Predictor - Streamlit web app.

An interactive clinical decision-support demo. A user enters patient metrics and
the app returns a calibrated risk probability from the trained model, plus the
factors most influencing the prediction.

Run locally:
    streamlit run app.py

Disclaimer: This is an educational project, NOT a medical device. It must not be
used for real diagnosis or treatment decisions.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data_prep import FEATURE_COLUMNS

MODEL_PATH = Path(__file__).resolve().parent / "models" / "heart_model.joblib"
METRICS_PATH = Path(__file__).resolve().parent / "models" / "metrics.json"

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
)


@st.cache_resource
def load_model():
    # Self-bootstrap: train the model on first run if it hasn't been built yet
    # (e.g. a fresh clone or a cloud deployment where the binary isn't committed).
    if not MODEL_PATH.exists():
        from src.train import main as train_main

        train_main()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


def risk_band(prob: float) -> tuple[str, str]:
    """Map a probability to a human-readable band and colour."""
    if prob < 0.33:
        return "Low", "green"
    if prob < 0.66:
        return "Moderate", "orange"
    return "High", "red"


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("❤️ Heart Disease Risk Predictor")
st.caption(
    "Enter patient clinical metrics to estimate the probability of coronary "
    "heart disease. Trained on the UCI Cleveland Heart Disease dataset."
)

model = load_model()
metrics = load_metrics()

# --------------------------------------------------------------------------- #
# Sidebar - model info
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Model performance")
    if metrics:
        st.metric("Test ROC-AUC", f"{metrics.get('test_roc_auc', 0):.3f}")
        st.metric("Test accuracy", f"{metrics.get('test_accuracy', 0):.1%}")
        st.metric("Test F1", f"{metrics.get('test_f1', 0):.3f}")
        st.caption(f"Best model: **{metrics.get('best_model', 'n/a')}**")
        st.caption(
            f"Trained on {metrics.get('n_train', '?')} patients, "
            f"tested on {metrics.get('n_test', '?')}."
        )
    st.divider()
    st.warning(
        "⚠️ Educational demo only. This is **not** a medical device and must "
        "not be used for real diagnosis."
    )

# --------------------------------------------------------------------------- #
# Input form
# --------------------------------------------------------------------------- #
st.subheader("Patient metrics")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 100, 54)
    sex = st.radio("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female", horizontal=True)
    cp = st.selectbox(
        "Chest pain type",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Typical angina",
            2: "Atypical angina",
            3: "Non-anginal pain",
            4: "Asymptomatic",
        }[x],
    )
    trestbps = st.slider("Resting blood pressure (mm Hg)", 80, 220, 130)
    chol = st.slider("Serum cholesterol (mg/dl)", 100, 600, 245)

with col2:
    fbs = st.radio(
        "Fasting blood sugar > 120 mg/dl",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
        horizontal=True,
    )
    restecg = st.selectbox(
        "Resting ECG",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Normal",
            1: "ST-T wave abnormality",
            2: "Left ventricular hypertrophy",
        }[x],
    )
    thalach = st.slider("Max heart rate achieved", 60, 220, 150)
    exang = st.radio(
        "Exercise-induced angina",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
        horizontal=True,
    )

with col3:
    oldpeak = st.slider("ST depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
    slope = st.selectbox(
        "Slope of peak exercise ST segment",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x],
    )
    ca = st.selectbox("Major vessels coloured by fluoroscopy", options=[0, 1, 2, 3])
    thal = st.selectbox(
        "Thalassemia",
        options=[3, 6, 7],
        format_func=lambda x: {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}[x],
    )

# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
if st.button("Predict risk", type="primary", use_container_width=True):
    row = pd.DataFrame(
        [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
        columns=FEATURE_COLUMNS,
    )
    prob = float(model.predict_proba(row)[0, 1])
    band, color = risk_band(prob)

    st.divider()
    left, right = st.columns([1, 2])
    with left:
        st.metric("Estimated risk", f"{prob:.1%}")
        st.markdown(f"### :{color}[{band} risk]")
    with right:
        st.progress(prob)
        if band == "High":
            st.error("High estimated risk. Clinical follow-up strongly recommended.")
        elif band == "Moderate":
            st.warning("Moderate estimated risk. Consider further evaluation.")
        else:
            st.success("Low estimated risk based on the provided metrics.")

    st.caption(
        "This probability reflects patterns in historical data only and is not a "
        "diagnosis. Always consult a qualified clinician."
    )
