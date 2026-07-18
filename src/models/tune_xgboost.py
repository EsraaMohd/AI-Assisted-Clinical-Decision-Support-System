from pathlib import Path
import joblib
from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split,
)

from xgboost import XGBClassifier

from src.data.load_data import load_dataset
from src.features.build_features import build_features

from src.models.evaluate_model import evaluate_model
from src.models.save_model import save_model

from src.analysis.feature_importance import (
    plot_feature_importance,
)
from src.analysis.threshold_analysis import threshold_analysis
from src.analysis.calibration_curve import (
    plot_calibration_curve,
)
"""
Hyperparameter Tuning
XGBoost Classifier
"""

DATA_PATH = Path(
    "data/raw/hospital_readmissions.csv"
)

MODEL_PATH = Path(
    "saved_models/xgboost_tuned.pkl"
)

FEATURE_IMPORTANCE_PATH = Path(
    "images/eda/xgboost_feature_importance.png"
)


def main():

    # -----------------------------
    # Load Dataset
    # -----------------------------

    df = load_dataset(DATA_PATH)

    # -----------------------------
    # Feature Engineering
    # -----------------------------

    df = build_features(df)

    # -----------------------------
    # Split Features & Target
    # -----------------------------

    X = df.drop(columns=["readmitted"])
    y = df["readmitted"]

    # -----------------------------
    # Train/Test Split
    # -----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # -----------------------------
    # Save Feature Columns
    # -----------------------------

    joblib.dump(
        X_train.columns.tolist(),
        "saved_models/feature_columns.pkl",
    )

    print("Feature columns saved successfully.")

    # -----------------------------
    # Base Model
    # -----------------------------

    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
    )

    # -----------------------------
    # Hyperparameter Search Space
    # -----------------------------

    param_dist = {

        "n_estimators": [100, 200, 300],

        "max_depth": [3, 5, 7],

        "learning_rate": [0.01, 0.05, 0.1],

        "subsample": [0.8, 1.0],

        "colsample_bytree": [0.8, 1.0],

    }

    # -----------------------------
    # Random Search
    # -----------------------------

    search = RandomizedSearchCV(

        estimator=model,

        param_distributions=param_dist,

        n_iter=25,

        scoring="recall",

        cv=5,

        random_state=42,

        verbose=2,

        n_jobs=-1,

    )

    print("\nStarting Hyperparameter Tuning...\n")

    search.fit(
        X_train,
        y_train,
    )

    print("\nTuning Finished!\n")

    print("=" * 60)
    print("Best Parameters")
    print("=" * 60)

    print(search.best_params_)

    print("\nBest CV Recall")

    print(
        f"{search.best_score_:.4f}"
    )

    # -----------------------------
    # Best Model
    # -----------------------------

    best_model = search.best_estimator_

    predictions = best_model.predict(
        X_test
    )

    probabilities = best_model.predict_proba(
        X_test
    )[:, 1]

    # -----------------------------
    # Evaluation
    # -----------------------------

    evaluate_model(

        y_true=y_test,

        y_pred=predictions,

        y_prob=probabilities,

    )
    threshold_analysis(

    y_true=y_test,

    probabilities=probabilities,

    save_plot_path=Path(
        "images/eda/threshold_analysis.png"
    ),

    save_csv_path=Path(
        "reports/threshold_results.csv"
    ),

    )

    plot_calibration_curve(

    y_true=y_test,

    y_prob=probabilities,

    save_path=Path(
        "images/eda/calibration_curve.png"
    ),

    )
    # -----------------------------
    # Feature Importance
    # -----------------------------

    plot_feature_importance(

        model=best_model,

        X=X_train,

        save_path=FEATURE_IMPORTANCE_PATH,

    )

    # -----------------------------
    # Save Model
    # -----------------------------

    save_model(

        model=best_model,

        file_path=MODEL_PATH,

    )


if __name__ == "__main__":
    main()