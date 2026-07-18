from pathlib import Path

import joblib


# ---------------------------------
# Paths
# ---------------------------------

MODEL_PATH = Path(
    "saved_models/xgboost_tuned.pkl"
)

FEATURE_COLUMNS_PATH = Path(
    "saved_models/feature_columns.pkl"
)


def predict_patient(
    df,
    threshold=0.30,
):
    """
    Predict hospital readmission risk
    for one patient.
    """

    # ---------------------------------
    # Load trained model
    # ---------------------------------

    model = joblib.load(
        MODEL_PATH
    )

    # ---------------------------------
    # Load feature columns
    # ---------------------------------

    feature_columns = joblib.load(
        FEATURE_COLUMNS_PATH
    )

    # ---------------------------------
    # Add missing columns
    # ---------------------------------

    for column in feature_columns:

        if column not in df.columns:

            df[column] = 0

    # ---------------------------------
    # Remove extra columns
    # ---------------------------------

    df = df[feature_columns]

    # ---------------------------------
    # Prediction
    # ---------------------------------

    prediction = model.predict(
        df
    )[0]

    probability = model.predict_proba(
        df
    )[0][1]

    # ---------------------------------
    # Risk Level
    # ---------------------------------

    if probability >= threshold:

        risk = "High Risk"

    else:

        risk = "Low Risk"

    # ---------------------------------
    # Return Results
    # ---------------------------------

    return {

        "prediction": int(prediction),

        "probability": float(probability),

        "risk": risk,

    }