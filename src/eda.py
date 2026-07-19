"""
Exploratory data analysis for the Heart Disease dataset.

Generates a summary figure (target balance, age distribution by outcome,
correlation heatmap) saved to reports/eda_summary.png.

Run:
    python -m src.eda
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_prep import get_clean_dataframe

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    df = get_clean_dataframe()

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Target balance
    counts = df["target"].map({0: "No disease", 1: "Disease"}).value_counts()
    axes[0].bar(counts.index, counts.values, color=["#22c55e", "#e11d48"])
    axes[0].set_title("Diagnosis balance")
    axes[0].set_ylabel("Patients")

    # 2. Age distribution by outcome
    for label, color in [(0, "#22c55e"), (1, "#e11d48")]:
        subset = df[df["target"] == label]["age"]
        axes[1].hist(subset, bins=15, alpha=0.6, color=color,
                     label="Disease" if label else "No disease")
    axes[1].set_title("Age distribution by outcome")
    axes[1].set_xlabel("Age")
    axes[1].legend()

    # 3. Correlation heatmap
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0,
                ax=axes[2], cbar_kws={"shrink": 0.7})
    axes[2].set_title("Feature correlation")

    fig.tight_layout()
    out = REPORTS_DIR / "eda_summary.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved EDA figure -> {out}")


if __name__ == "__main__":
    main()
