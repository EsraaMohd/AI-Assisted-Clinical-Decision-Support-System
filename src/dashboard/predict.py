from pathlib import Path

import joblib
import shap


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
    and generate a local SHAP explanation
    for a single patient.
    """

    # ---------------------------------
    # Load trained model
    # ---------------------------------

    model = joblib.load(
        MODEL_PATH
    )

    # ---------------------------------
    # Load training feature columns
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

    df = df[
        feature_columns
    ]

    # ---------------------------------
    # Prediction
    # ---------------------------------

    probability = model.predict_proba(
        df
    )[0][1]

    # ---------------------------------
    # Threshold-based decision
    # ---------------------------------

    if probability >= threshold:

        prediction = 1

        risk = "High Risk"

    else:

        prediction = 0

        risk = "Low Risk"

    # ---------------------------------
    # SHAP Explanation
    # ---------------------------------

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        df
    )

    # ---------------------------------
    # Extract SHAP values
    # for this patient
    # ---------------------------------

    patient_shap_values = shap_values[0]

    # ---------------------------------
    # Create explanation DataFrame
    # ---------------------------------

    explanation = []

    for feature, value, shap_value in zip(

        df.columns,

        df.iloc[0].values,

        patient_shap_values,

    ):

        explanation.append(

            {

                "feature": feature,

                "value": value,

                "shap_value": shap_value,

            }

        )

    # ---------------------------------
    # Sort by absolute influence
    # ---------------------------------

    explanation = sorted(

        explanation,

        key=lambda x: abs(

            x["shap_value"]

        ),

        reverse=True,

    )

    # ---------------------------------
    # Return results
    # ---------------------------------

    return {

        "prediction": prediction,

        "probability": float(

            probability

        ),

        "risk": risk,

        "shap_explanation": explanation,

    }