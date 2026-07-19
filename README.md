# ❤️ Heart Disease Risk Predictor

An end-to-end machine learning project that estimates a patient's risk of coronary heart disease from routine clinical metrics. It covers the full data-science lifecycle — **data cleaning → EDA → model training & selection → evaluation → an interactive web app** — and is built to run locally in VS Code and deploy anywhere.

> ⚠️ **Disclaimer:** This is an **educational project**, not a medical device. It must not be used for real diagnosis or treatment decisions.

---

## 🎯 The real-world problem

Cardiovascular disease is the leading cause of death worldwide. Early risk stratification helps clinicians prioritise patients for further testing. This project trains a model on real anonymised patient data and exposes it through a simple, interpretable web interface a non-technical user can operate.

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository — Heart Disease Data Set](https://archive.ics.uci.edu/dataset/45/heart+disease) (Cleveland subset)
- **Rows:** 303 patients
- **Features:** 13 clinical attributes (age, sex, chest pain type, blood pressure, cholesterol, max heart rate, ST depression, etc.)
- **Target:** presence of heart disease (binarised: 0 = absent, 1 = present)

## 🤖 Model performance

Three classifiers were compared with 5-fold cross-validated ROC-AUC; **Logistic Regression** won and was evaluated on a held-out 20% test set.

| Metric (held-out test set) | Score |
|---|---|
| ROC-AUC | **0.951** |
| Accuracy | **86.9%** |
| F1 score | **0.867** |

| Model | CV ROC-AUC |
|---|---|
| Logistic Regression ✅ | 0.896 |
| Random Forest | 0.883 |
| Gradient Boosting | 0.842 |

Every model is wrapped in a scikit-learn `Pipeline` (StandardScaler + classifier) so preprocessing is applied consistently at inference time.

## 🗂️ Project structure

```
heart-disease-predictor/
├── app.py                  # Streamlit web app (frontend)
├── requirements.txt
├── LICENSE
├── data/
│   ├── processed.cleveland.data   # raw UCI data
│   └── heart_clean.csv            # cleaned dataset (generated)
├── models/
│   ├── heart_model.joblib         # trained pipeline (generated)
│   └── metrics.json               # evaluation report (generated)
├── reports/
│   └── eda_summary.png            # EDA figure (generated)
├── src/
│   ├── data_prep.py        # load + clean data
│   ├── eda.py              # exploratory analysis figure
│   └── train.py           # train, select, evaluate, persist model
└── tests/
    └── test_pipeline.py    # pytest unit tests
```

## 🚀 Getting started (VS Code)

```bash
# 1. Clone
git clone https://github.com/<your-username>/heart-disease-predictor.git
cd heart-disease-predictor

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Re-run the full pipeline
python -m src.data_prep     # clean the raw data
python -m src.eda          # generate EDA figure
python -m src.train        # train + evaluate + save the model

# 5. Launch the web app
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## 🧪 Running the tests

```bash
python -m pytest -q
```

## 🌐 Deployment

The app is a standard Streamlit application and can be deployed for free on
[Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this repo to GitHub.
2. Go to Streamlit Community Cloud → **New app** → point it at this repo and `app.py`.
3. Deploy. On first launch the app **auto-trains the model** from the committed
   dataset if `models/heart_model.joblib` isn't present, so no manual training
   step is required. (Training takes a couple of seconds.)

## 🛠️ Tech stack

- **Python** · **scikit-learn** (modelling) · **pandas / numpy** (data)
- **matplotlib / seaborn** (EDA) · **Streamlit** (frontend) · **pytest** (testing)

## 📄 License

Released under the [MIT License](LICENSE).
