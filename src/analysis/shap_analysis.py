from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from sklearn.model_selection import train_test_split

from src.data.load_data import load_dataset
from src.features.build_features import build_features


# =====================================================
# PATHS
# =====================================================

DATA_PATH = Path(
    "data/raw/hospital_readmissions.csv"
)

MODEL_PATH = Path(
    "saved_models/xgboost_tuned.pkl"
)

OUTPUT_PATH = Path(
    "images/eda/shap_summary.png"
)


# =====================================================
# MAIN
# =====================================================

def main():

    # -------------------------------------------------
    # Load Dataset
    # -------------------------------------------------

    df = load_dataset(
        DATA_PATH
    )

    # -------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------

    df = build_features(
        df
    )

    # -------------------------------------------------
    # Split Features and Target
    # -------------------------------------------------

    X = df.drop(
        columns=["readmitted"]
    )

    y = df["readmitted"]

    # -------------------------------------------------
    # Same Train/Test Split
    # -------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y,

    )

    # -------------------------------------------------
    # Load Trained Model
    # -------------------------------------------------

    model = joblib.load(

        MODEL_PATH

    )

    # -------------------------------------------------
    # Create SHAP Explainer
    # -------------------------------------------------

    explainer = shap.TreeExplainer(

        model

    )

    # -------------------------------------------------
    # Calculate SHAP Values
    # -------------------------------------------------

    shap_values = explainer.shap_values(

        X_test

    )

    # -------------------------------------------------
    # Create Output Directory
    # -------------------------------------------------

    OUTPUT_PATH.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    # -------------------------------------------------
    # SHAP Summary Plot
    # -------------------------------------------------

    plt.figure(

        figsize=(12, 8)

    )

    shap.summary_plot(

        shap_values,

        X_test,

        show=False,

    )

    plt.tight_layout()

    plt.savefig(

        OUTPUT_PATH,

        dpi=300,

        bbox_inches="tight",

    )

    plt.close()

    print(

        "\nSHAP analysis completed successfully."

    )

    print(

        f"Saved to: {OUTPUT_PATH}"

    )


if __name__ == "__main__":

    main()