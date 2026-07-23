from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from src.data.load_data import load_dataset
from src.features.build_features import build_features


DATA_PATH = Path(
    "data/raw/hospital_readmissions.csv"
)

MODEL_PATH = Path(
    "saved_models/xgboost_tuned.pkl"
)

FEATURE_COLUMNS_PATH = Path(
    "saved_models/feature_columns.pkl"
)

OUTPUT_PATH = Path(
    "reports/confusion_matrix_threshold_030.csv"
)


def main():

    # ---------------------------------------------
    # Load Dataset
    # ---------------------------------------------

    df = load_dataset(DATA_PATH)

    # ---------------------------------------------
    # Feature Engineering
    # ---------------------------------------------

    df = build_features(df)

    # ---------------------------------------------
    # Split Features and Target
    # ---------------------------------------------

    X = df.drop(
        columns=["readmitted"]
    )

    y = df["readmitted"]

    # ---------------------------------------------
    # Same Test Split Used During Training
    # ---------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y,

    )

    # ---------------------------------------------
    # Load Model
    # ---------------------------------------------

    model = joblib.load(
        MODEL_PATH
    )

    # ---------------------------------------------
    # Predict Probabilities
    # ---------------------------------------------

    probabilities = model.predict_proba(

        X_test

    )[:, 1]

    # ---------------------------------------------
    # Apply Selected Threshold
    # ---------------------------------------------

    threshold = 0.30

    predictions = (

        probabilities >= threshold

    ).astype(int)

    # ---------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------

    tn, fp, fn, tp = confusion_matrix(

        y_test,

        predictions,

    ).ravel()

    # ---------------------------------------------
    # Save Results
    # ---------------------------------------------

    result = pd.DataFrame(

        {

            "Actual": [

                "No Readmission",

                "No Readmission",

                "Readmission",

                "Readmission",

            ],

            "Predicted": [

                "No Readmission",

                "Readmission",

                "No Readmission",

                "Readmission",

            ],

            "Count": [

                tn,

                fp,

                fn,

                tp,

            ],

        }

    )

    OUTPUT_PATH.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    result.to_csv(

        OUTPUT_PATH,

        index=False,

    )

    print(

        "\nConfusion matrix saved successfully:"

    )

    print(

        OUTPUT_PATH

    )

    print(

        "\n"

    )

    print(result)


if __name__ == "__main__":

    main()