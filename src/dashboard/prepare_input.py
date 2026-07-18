import pandas as pd

from src.features.build_features import build_features


def prepare_input(patient_data):
    """
    Convert user input dictionary
    into the same feature format
    used during model training.
    """

    # -----------------------------
    # Dictionary → DataFrame
    # -----------------------------

    df = pd.DataFrame(
        [patient_data]
    )

    # -----------------------------
    # Apply Feature Engineering
    # -----------------------------

    df = build_features(df)

    return df