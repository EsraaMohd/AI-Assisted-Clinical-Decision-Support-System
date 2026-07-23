from pathlib import Path

import joblib


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
    for a single patient.
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
    # Keep training column order
    # ---------------------------------

    df = df[feature_columns]

    # ---------------------------------
    # Probability Prediction
    # ---------------------------------

    probability = model.predict_proba(
        df
    )[0][1]

    # ---------------------------------
    # Apply Optimized Threshold
    # ---------------------------------

    prediction = int(
        probability >= threshold
    )

    # ---------------------------------
    # Risk Level
    # ---------------------------------

    if prediction == 1:

        risk = "High Risk"

    else:

        risk = "Low Risk"

    # ---------------------------------
    # Return Results
    # ---------------------------------

    return {

        "prediction": prediction,

        "probability": float(
            probability
        ),

        "risk": risk,

        "threshold": threshold,

    }