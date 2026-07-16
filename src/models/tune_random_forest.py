from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
)

from src.data.load_data import load_dataset
from src.features.build_features import build_features
from src.features.split_features import split_features_target
from src.models.evaluate_model import evaluate_model
from src.models.save_model import save_model

from pathlib import Path

"""
Hyperparameter Tuning for Random Forest
"""

DATA_PATH = Path("data/raw/hospital_readmissions.csv")

MODEL_PATH = Path(
    "saved_models/random_forest_tuned.pkl"
)


def main():

    # -----------------------------
    # Load Dataset
    # -----------------------------

    df = load_dataset(DATA_PATH)

    # -----------------------------
    # Feature Engineering
    # -----------------------------

    processed_df = build_features(df)

    # -----------------------------
    # Split Features & Target
    # -----------------------------

    X, y = split_features_target(processed_df)

    # -----------------------------
    # Train Test Split
    # -----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # -----------------------------
    # Base Model
    # -----------------------------

    model = RandomForestClassifier(
        random_state=42,
    )

    # -----------------------------
    # Hyperparameter Search Space
    # -----------------------------

    param_grid = {

        "n_estimators": [100, 200],

        "max_depth": [
            10,
            None,
        ],

        "min_samples_split": [
            2,
            5,
        ],

        "min_samples_leaf": [
            1,
            2,
        ],

    }

    # -----------------------------
    # Grid Search
    # -----------------------------

    grid_search = GridSearchCV(

        estimator=model,

        param_grid=param_grid,

        scoring="recall",

        cv=5,

        n_jobs=-1,

        verbose=2,

    )

    print("\nStarting Hyperparameter Tuning...\n")

    grid_search.fit(
        X_train,
        y_train,
    )

    print("\nTuning Finished!\n")

    # -----------------------------
    # Best Parameters
    # -----------------------------

    print("=" * 50)

    print("Best Parameters")

    print("=" * 50)

    print(grid_search.best_params_)

    print()

    print(f"Best CV Recall: {grid_search.best_score_:.4f}")

    # -----------------------------
    # Best Model
    # -----------------------------

    best_model = grid_search.best_estimator_

    # -----------------------------
    # Test Set Evaluation
    # -----------------------------

    predictions = best_model.predict(
        X_test,
    )

    probabilities = best_model.predict_proba(
        X_test,
    )[:, 1]

    evaluate_model(

        y_true=y_test,

        y_pred=predictions,

        y_prob=probabilities,

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