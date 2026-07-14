import pandas as pd


def split_features_target(df: pd.DataFrame):
    """
    Split the processed dataframe into features and target.

    Args:
        df: Processed dataframe.

    Returns:
        X: Feature matrix.
        y: Target vector.
    """

    X = df.drop(columns=["readmitted"])

    y = df["readmitted"]

    return X, y