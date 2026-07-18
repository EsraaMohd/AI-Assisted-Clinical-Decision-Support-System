from xml.parsers.expat import model

import joblib
from xgboost import XGBClassifier
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.analysis.feature_importance import plot_feature_importance
from src.data.load_data import load_dataset
from src.features.build_features import build_features
from src.models.evaluate_model import evaluate_model
from src.models.save_model import save_model

"""
Train Random Forest Classifier
"""


DATA_PATH = Path("data/raw/hospital_readmissions.csv")

MODEL_PATH = Path("saved_models/xgboost.pkl")

FEATURE_IMPORTANCE_PATH = Path(
    "images/eda/feature_importance.png"
)


def main():

    # -----------------------------
    # Load dataset
    # -----------------------------

    df = load_dataset(DATA_PATH)

    # -----------------------------
    # Feature Engineering
    # -----------------------------

    df = build_features(df)

    # -----------------------------
    # Split Features & Target
    # -----------------------------

    X = df.drop(
        columns=["readmitted"]
    )

    y = df["readmitted"]

    # -----------------------------
    # Train / Test Split
    # -----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nDataset Shapes")

    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")

    print(f"y_train : {y_train.shape}")
    print(f"y_test  : {y_test.shape}")

    # -----------------------------
    # Build Model
    # -----------------------------

    model = XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    )

    # -----------------------------
    # Train
    # -----------------------------

    model.fit(
        X_train,
        y_train,
    )

   

    # -----------------------------
    # Predictions
    # -----------------------------

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:,1]
    # -----------------------------
    # Evaluation
    # -----------------------------

    evaluate_model(
        y_true=y_test,
        y_pred=predictions,
        y_prob=probabilities,
    )

    # -----------------------------
    # Feature Importance
    # -----------------------------

    plot_feature_importance(
        model=model,
        X=X_train,
        save_path=FEATURE_IMPORTANCE_PATH,
    )

    # -----------------------------
    # Save Model
    # -----------------------------

    save_model(
        model=model,
        file_path=MODEL_PATH,
    )


if __name__ == "__main__":
    main()